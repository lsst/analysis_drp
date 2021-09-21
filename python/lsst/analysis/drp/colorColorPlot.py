import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import MagColumnNanoJansky
from .calcFunctors import MagDiff
from .plotUtils import parsePlotInfo, addPlotInfo
from . import dataSelectors as dataSelectors


class ColorColorPlotTaskConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("tract", "skymap"),
                                    defaultTemplates={"inputCoaddName": "deep", "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="objectTable_tract",
                                             dimensions=("tract", "skymap"),
                                             deferLoad=True)

    colorColorPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                     storageClass="Plot",
                                                     name="colorColorPlot_{plotName}",
                                                     dimensions=("tract", "skymap"))


class ColorColorPlotTaskConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ColorColorPlotTaskConnections):

    axisActions = ConfigurableActionStructField(
        doc="The actions to use to calculate the values used on each axis.",
        default={"xAction": MagDiff, "yAction": MagDiff, "zAction": MagColumnNanoJansky},
    )

    axisLabels = pexConfig.DictField(
        doc="Name of the column in the dataframe to plot, will be used as axis label: {'x':, 'y':, 'z':}",
        keytype=str,
        itemtype=str,
    )

    sourceIdentifierActions = ConfigurableActionStructField(
        doc="What types of sources to use.",
        default={"starIdentifier": dataSelectors.StarIdentifier(),
                 "galaxyIdentifier": dataSelectors.GalaxyIdentifier()},
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": dataSelectors.CoaddPlotFlagSelector},
    )


class ColorColorPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorPlotTaskConfig
    _DefaultName = "ColorColorPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        columnNames = set(["patchId"])
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.sourceIdentifierActions]:
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)

        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs["catPlot"] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, tableName):

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask]

        columns = {self.config.axisLabels["x"]: self.config.axisActions.xAction(catPlot),
                   self.config.axisLabels["y"]: self.config.axisActions.yAction(catPlot),
                   self.config.axisLabels["z"]: self.config.axisActions.zAction(catPlot),
                   "patchId": catPlot["patchId"]}
        plotDf = pd.DataFrame(columns)

        sourceTypes = np.zeros(len(plotDf))
        for identifier in self.config.sourceIdentifierActions:
            # The source identifiers return 1 for a star and 2 for a galaxy
            # rather than a mask this allows the information about which
            # type of sources are being plotted to be propagated
            sourceTypes += identifier(catPlot)
        # If the plot requires all the sources then assign 10 to
        # signify this.
        if list(self.config.sourceIdentifierActions) == []:
            sourceTypes = [10]*len(plotDf)
        plotDf.loc[:, "sourceType"] = sourceTypes

        # Check the columns have finite values
        mask = np.ones(len(catPlot), dtype=bool)
        for col in plotDf.columns:
            mask &= np.isfinite(plotDf[col])
        plotDf = plotDf[mask]

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName)
        # Make the plot
        fig = self.colorColorPlot(plotDf, plotInfo)

        return pipeBase.Struct(colorColorPlot=fig)

    def colorColorPlot(self, catPlot, plotInfo):
        """Makes color-color plots

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        plotInfo : `dict`
            A dictionary of information about the data being plotted with keys:
                ``"run"``
                    The output run for the plots (`str`).
                ``"filter"``
                    The filter used for this data (`str`).
                ``"tract"``
                    The tract that the data comes from (`str`).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        Makes a color-color plot of `self.config.axisLabels['x']` against
        `self.config.axisLabels['y']`, these points are color coded by i band
        CModel magnitude. The axis labels are given by `self.config.xLabel`
        and `self.config.yLabel`. The column given in
        `self.config.sourceTypeColName` is used for star galaxy separation.
        """

        self.log.info(("Plotting {}: {} against {} on a color-color plot.".format(
                       self.config.connections.plotName, self.config.axisLabels["x"],
                       self.config.axisLabels["y"])))

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.11, 0.65, 0.75])

        # Need to separate stars and galaxies
        stars = (catPlot["sourceType"] == 1)
        galaxies = (catPlot["sourceType"] == 2)

        xCol = self.config.axisLabels["x"]
        yCol = self.config.axisLabels["y"]
        zCol = self.config.axisLabels["z"]

        # For galaxies
        xsGalaxies = catPlot.loc[galaxies, xCol]
        ysGalaxies = catPlot.loc[galaxies, yCol]
        zsGalaxies = catPlot.loc[galaxies, zCol]

        # For stars
        xsStars = catPlot.loc[stars, xCol]
        ysStars = catPlot.loc[stars, yCol]
        zsStars = catPlot.loc[stars, zCol]

        [vminGals, vmaxGals] = np.nanpercentile(zsGalaxies, [1, 99])
        [vminStars, vmaxStars] = np.nanpercentile(zsStars, [1, 99])
        galPoints = ax.scatter(xsGalaxies, ysGalaxies, c=zsGalaxies, cmap="autumn_r", label="Galaxies",
                               s=0.5, vmin=vminGals, vmax=vmaxGals)
        starPoints = ax.scatter(xsStars, ysStars, c=zsStars, cmap="winter_r", label="Stars", s=0.5,
                                vmin=vminStars, vmax=vmaxStars)

        # Add text details
        fig.text(0.70, 0.9, "Num. Galaxies: {}".format(galaxies.sum()), color="C1")
        fig.text(0.70, 0.93, "Num. Stars: {}".format(stars.sum()), color="C0")

        # Add colorbars
        galCbAx = fig.add_axes([0.85, 0.11, 0.04, 0.75])
        plt.colorbar(galPoints, cax=galCbAx, extend="both")
        galCbAx.yaxis.set_ticks_position("left")
        starCbAx = fig.add_axes([0.89, 0.11, 0.04, 0.75])
        plt.colorbar(starPoints, cax=starCbAx, extend="both")
        magLabel = self.config.axisLabels["z"]
        galCbAx.text(0.6, 0.5, magLabel + ": Galaxies", color="k", rotation="vertical",
                     transform=galCbAx.transAxes, ha="center", va="center", fontsize=10)
        starCbAx.text(0.6, 0.5, magLabel + ": Stars", color="k", rotation="vertical",
                      transform=starCbAx.transAxes, ha="center", va="center", fontsize=10)

        ax.set_xlabel(self.config.axisLabels["x"])
        ax.set_ylabel(self.config.axisLabels["y"])

        # Set useful axis limits
        starPercsX = np.nanpercentile(xsStars, [1, 99.5])
        starPercsY = np.nanpercentile(ysStars, [1, 99.5])
        ax.set_xlim(starPercsX[0], starPercsX[1])
        ax.set_ylim(starPercsY[0], starPercsY[1])

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
