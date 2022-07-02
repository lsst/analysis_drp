import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pathEffects
import numpy as np
import pandas as pd

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import MagColumnNanoJansky
from .calcFunctors import ExtinctionCorrectedMagDiff
from .plotUtils import parsePlotInfo, addPlotInfo, mkColormap
from . import dataSelectors as dataSelectors

matplotlib.use("Agg")


class ColorColorPlotConnections(pipeBase.PipelineTaskConnections,
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


class ColorColorPlotConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=ColorColorPlotConnections):

    axisActions = ConfigurableActionStructField(
        doc="The actions to use to calculate the values used on each axis.",
        default={"xAction": ExtinctionCorrectedMagDiff, "yAction": ExtinctionCorrectedMagDiff,
                 "zAction": MagColumnNanoJansky},
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
        default={"flagSelector": dataSelectors.CoaddPlotFlagSelector,
                 "catSnSelector": dataSelectors.SnSelector},
    )


class ColorColorPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorPlotConfig
    _DefaultName = "colorColorPlot"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        columnNames = set(["patch"])
        bands = []
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.sourceIdentifierActions]:
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)
                    band = col.split("_")[0]
                    if band not in ["coord", "extend", "detect", "xy", "merge", "ebv", "sky"]:
                        bands.append(band)

        bands = set(bands)
        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs["catPlot"] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        inputs["plotName"] = localConnections.colorColorPlot.name
        inputs["bands"] = bands
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, tableName, bands, plotName):

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask]

        columns = {self.config.axisLabels["x"]: self.config.axisActions.xAction(catPlot),
                   self.config.axisLabels["y"]: self.config.axisActions.yAction(catPlot),
                   self.config.axisLabels["z"]: self.config.axisActions.zAction(catPlot),
                   "patch": catPlot["patch"]}
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

        # Get the S/N cut used (if any)
        if hasattr(self.config.selectorActions, "catSnSelector"):
            SN = self.config.selectorActions.catSnSelector.threshold
            SNFlux = self.config.selectorActions.catSnSelector.fluxType
        else:
            SN = "N/A"
            SNFlux = "N/A"

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, plotName, SN, SNFlux)
        # Make the plot
        if len(plotDf) == 0:
            fig = plt.Figure()
            noDataText = ("No data to plot after selectors applied\n(do you have all three of "
                          "the bands required: {}?)".format(bands))
            fig.text(0.5, 0.5, noDataText, ha="center", va="center")
            fig = addPlotInfo(fig, plotInfo)
        else:
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

        self.log.info(("Plotting %s: %s against %s on a color-color plot.",
                       self.config.connections.plotName, self.config.axisLabels["x"],
                       self.config.axisLabels["y"]))

        # Define a new colormap
        newBlues = mkColormap(["paleturquoise", "midnightblue"])
        newReds = mkColormap(["lemonchiffon", "firebrick"])

        fig = plt.figure(dpi=300)
        ax = fig.add_axes([0.15, 0.11, 0.65, 0.75])

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

        if len(zsGalaxies) > 0:
            [vminGals, vmaxGals] = np.nanpercentile(zsGalaxies, [1, 99])
            galPoints = ax.scatter(xsGalaxies, ysGalaxies, c=zsGalaxies, cmap=newReds, label="Galaxies",
                                   s=0.5, vmin=vminGals, vmax=vmaxGals)
        else:
            galPoints = None

        if len(zsStars) > 0:
            [vminStars, vmaxStars] = np.nanpercentile(zsStars, [1, 99])
            starPoints = ax.scatter(xsStars, ysStars, c=zsStars, cmap=newBlues, label="Stars", s=0.5,
                                    vmin=vminStars, vmax=vmaxStars)
        else:
            starPoints = None

        # Add text details
        galBBox = dict(facecolor="lemonchiffon", alpha=0.5, edgecolor="none")
        fig.text(0.70, 0.9, "Num. Galaxies: {}".format(galaxies.sum()), bbox=galBBox, fontsize=8)
        starBBox = dict(facecolor="paleturquoise", alpha=0.5, edgecolor="none")
        fig.text(0.70, 0.96, "Num. Stars: {}".format(stars.sum()), bbox=starBBox, fontsize=8)

        # Add colorbars
        magLabel = self.config.axisLabels["z"]
        if galPoints:
            galCbAx = fig.add_axes([0.85, 0.11, 0.04, 0.75])
            plt.colorbar(galPoints, cax=galCbAx, extend="both")
            galCbAx.yaxis.set_ticks_position("left")
            galText = galCbAx.text(0.5, 0.5, magLabel + ": Galaxies", color="k", rotation="vertical",
                                   transform=galCbAx.transAxes, ha="center", va="center", fontsize=10)
            galText.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
        if starPoints:
            starCbAx = fig.add_axes([0.89, 0.11, 0.04, 0.75])
            plt.colorbar(starPoints, cax=starCbAx, extend="both")
            starText = starCbAx.text(0.5, 0.5, magLabel + ": Stars", color="k", rotation="vertical",
                                     transform=starCbAx.transAxes, ha="center", va="center", fontsize=10)
            starText.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

        ax.set_xlabel(self.config.axisLabels["x"])
        ax.set_ylabel(self.config.axisLabels["y"])

        # Set useful axis limits
        if len(xsStars) > 0:
            starPercsX = np.nanpercentile(xsStars, [1, 99.5])
            starPercsY = np.nanpercentile(ysStars, [1, 99.5])
            pad = (starPercsX[1] - starPercsX[0])/10
            ax.set_xlim(starPercsX[0] - pad, starPercsX[1] + pad)
            ax.set_ylim(starPercsY[0] - pad, starPercsY[1] + pad)

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
