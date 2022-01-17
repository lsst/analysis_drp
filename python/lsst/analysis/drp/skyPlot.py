import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pathEffects
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import CoordColumn, SingleColumnAction
from lsst.skymap import BaseSkyMap

from . import dataSelectors as dataSelectors
from .plotUtils import generateSummaryStats, parsePlotInfo, addPlotInfo

import pandas as pd

__all__ = ["SkyPlotTaskConfig", "SkyPlotTask"]


class SkyPlotTaskConnections(pipeBase.PipelineTaskConnections, dimensions=("tract", "skymap"),
                             defaultTemplates={"inputCoaddName": "deep", "plotName": "deltaCoords",
                                               "tableType": "forced"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract-wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="objectTable_tract",
                                             dimensions=("tract", "skymap"),
                                             deferLoad=True)

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
                                            dimensions=("skymap",))

    skyPlot = pipeBase.connectionTypes.Output(doc="A plot showing the on-sky distribution of a value.",
                                              storageClass="Plot",
                                              name="skyPlot_{plotName}",
                                              dimensions=("tract", "skymap"))


class SkyPlotTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=SkyPlotTaskConnections):

    axisActions = ConfigurableActionStructField(
        doc="The actions to use to calculate the values used on each axis. Used if <axis> in axisColNames "
            "is set to 'Functor'.",
        default={"xAction": CoordColumn, "yAction": CoordColumn, "zAction": SingleColumnAction},
    )

    axisLabels = pexConfig.DictField(
        doc="Name of column in dataframe to plot, will be used as axis label: {'x':, 'y':, 'z':}",
        keytype=str,
        itemtype=str,
    )

    sourceSelectorActions = ConfigurableActionStructField(
        doc="What types of sources to use.",
        default={"sourceSelector": dataSelectors.StarIdentifier},
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": dataSelectors.CoaddPlotFlagSelector},
    )

    statisticSelectorActions = ConfigurableActionStructField(
        doc="Selectors to use to decide which points to use for calculating statistics.",
        default={"statSelector": dataSelectors.SnSelector},
    )

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.xAction.column = "coord_ra"
        self.axisActions.xAction.inRadians = False
        self.axisActions.yAction.column = "coord_dec"
        self.axisActions.yAction.inRadians = False


class SkyPlotTask(pipeBase.PipelineTask):

    ConfigClass = SkyPlotTaskConfig
    _DefaultName = "skyPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        columnNames = set(["patch"])
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.statisticSelectorActions, self.config.sourceSelectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)

        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs['catPlot'] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, skymap, tableName):
        """Prep the catalogue and then make a skyPlot of the given column.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        dataId :
        `lsst.daf.butler.core.dimensions._coordinate._ExpandedTupleDataCoordinate`
            The dimensions that the plot is being made from.
        runName : `str`
            The name of the collection that the plot is written out to.
        skymap : `lsst.skymap`
            The skymap used to define the patch boundaries.
        tableName : `str`
            The type of table used to make the plot.

        Returns
        -------
        `pipeBase.Struct` containing:
            skyPlot : `matplotlib.figure.Figure`
                The resulting figure.

        Notes
        -----
        The catalogue is first narrowed down using the selectors specified in
        `self.config.selectorActions`.
        If the column names are 'Functor' then the functors specified in
        `self.config.axisFunctors` are used to calculate the required values.
        After this the following functions are run:

        `parsePlotInfo` which uses the dataId, runName and tableName to add
        useful information to the plot.

        `generateSummaryStats` which parses the skymap to give the corners of
        the patches for later plotting and calculates some basic statistics
        in each patch for the column in self.config.axisActions['zAction'].

        `SkyPlot` which makes the plot of the sky distribution of
        `self.config.axisActions['zAction']`.

        """

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
        for selector in self.config.sourceSelectorActions:
            # The source selectors return 1 for a star and 2 for a galaxy
            # rather than a mask this allows the information about which
            # type of sources are being plotted to be propagated
            sourceTypes += selector(catPlot)
        if list(self.config.sourceSelectorActions) == []:
            sourceTypes = [10]*len(plotDf)
        plotDf.loc[:, "sourceType"] = sourceTypes

        # Decide which points to use for stats calculation
        plotDf.loc[:, "useForStats"] = self.config.statisticSelectorActions.statSelector(catPlot)

        # Check the columns have finite values
        mask = np.ones(len(catPlot), dtype=bool)
        for col in plotDf.columns:
            mask &= np.isfinite(plotDf[col])
        plotDf = plotDf[mask]

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName)
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStats(plotDf, self.config.axisLabels["z"], skymap, plotInfo)
        # Make the plot
        fig = self.skyPlot(plotDf, plotInfo, sumStats)

        return pipeBase.Struct(skyPlot=fig)

    def skyPlot(self, catPlot, plotInfo, sumStats):
        """Makes a generic plot showing the value at given points on the sky.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        plotInfo : `dict`
            A dictionary of information about the data being plotted with keys:
                ``"run"``
                    The output run for the plots (`str`).
                ``"skymap"``
                    The type of skymap used for the data (`str`).
                ``"filter"``
                    The filter used for this data (`str`).
                ``"tract"``
                    The tract that the data comes from (`str`).
        sumStats : `dict`
            A dictionary where the patchIds are the keys which store the R.A.
            and dec of the corners of the patch.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        Uses the config options `self.config.xColName` and
        `self.config.yColName` to plot points color coded by
        `self.config.axisActions['zAction']`.
        The points plotted are those selected by the selectors specified in
        `self.config.selectorActions`.
        """
        self.log.info("Plotting {}: the values of {} on a sky plot.".format(
                      self.config.connections.plotName, self.config.axisLabels["z"]))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Need to separate stars and galaxies
        stars = (catPlot["sourceType"] == 1)
        galaxies = (catPlot["sourceType"] == 2)

        xCol = self.config.axisLabels["x"]
        yCol = self.config.axisLabels["y"]
        zCol = self.config.axisLabels["z"]

        # For galaxies
        xsGalaxies = catPlot.loc[galaxies, xCol]
        ysGalaxies = catPlot.loc[galaxies, yCol]
        colorValsGalaxies = catPlot.loc[galaxies, zCol]

        # For stars
        xsStars = catPlot.loc[stars, xCol]
        ysStars = catPlot.loc[stars, yCol]
        colorValsStars = catPlot.loc[stars, zCol]

        # Calculate some statistics
        if np.any(catPlot["sourceType"] == 2):
            statGals = ((catPlot["useForStats"] == 1) & galaxies)
            statGalMed = np.nanmedian(catPlot.loc[statGals, zCol])
            statGalMad = sigmaMad(catPlot.loc[statGals, zCol], nan_policy="omit")

            galStatsText = ("Median: {:0.2f}\n".format(statGalMed) + r"$\sigma_{MAD}$: "
                            + "{:0.2f}\n".format(statGalMad) + r"n$_{points}$: "
                            + "{}".format(len(xsGalaxies)))
            # Add statistics
            bbox = dict(facecolor="C1", alpha=0.3, edgecolor="none")
            ax.text(0.63, 0.91, galStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if np.any(catPlot["sourceType"] == 1):

            statStars = ((catPlot["useForStats"] == 1) & stars)
            statStarMed = np.nanmedian(catPlot.loc[statStars, zCol])
            statStarMad = sigmaMad(catPlot.loc[statStars, zCol], nan_policy="omit")

            starStatsText = ("Median: {:0.2f}\n".format(statStarMed) + r"$\sigma_{MAD}$: "
                             + "{:0.2f}\n".format(statStarMad) + r"n$_{points}$: "
                             + "{}".format(len(xsStars)))
            # Add statistics
            bbox = dict(facecolor="C0", alpha=0.3, edgecolor="none")
            ax.text(0.8, 0.91, starStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if np.any(catPlot["sourceType"] == 10):

            statAll = (catPlot["useForStats"] == 1)
            statAllMed = np.nanmedian(catPlot.loc[statAll, zCol])
            statAllMad = sigmaMad(catPlot.loc[statAll, zCol], nan_policy="omit")

            allStatsText = ("Median: {:0.2f}\n".format(statAllMed) + r"$\sigma_{MAD}$: "
                            + "{:0.2f}\n".format(statAllMad) + r"n$_{points}$: "
                            + "{}".format(len(catPlot)))
            bbox = dict(facecolor="purple", alpha=0.2, edgecolor="none")
            ax.text(0.8, 0.91, allStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if np.any(catPlot["sourceType"] == 9):

            statAll = (catPlot["useForStats"] == 1)
            statAllMed = np.nanmedian(catPlot.loc[statAll, zCol])
            statAllMad = sigmaMad(catPlot.loc[statAll, zCol], nan_policy="omit")

            allStatsText = ("Median: {:0.2f}\n".format(statAllMed) + r"$\sigma_{MAD}$: "
                            + "{:0.2f}\n".format(statAllMad) + r"n$_{points}$: "
                            + "{}".format(len(catPlot)))
            bbox = dict(facecolor="green", alpha=0.2, edgecolor="none")
            ax.text(0.8, 0.91, allStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        toPlotList = []
        if np.any(catPlot["sourceType"] == 1):
            toPlotList.append((xsStars, ysStars, colorValsStars, "winter_r", "Stars"))
        if np.any(catPlot["sourceType"] == 2):
            toPlotList.append((xsGalaxies, ysGalaxies, colorValsGalaxies, "autumn_r", "Galaxies"))
        if np.any(catPlot["sourceType"] == 10):
            toPlotList.append((catPlot[xCol], catPlot[yCol], catPlot[zCol], "plasma", "All"))
        if np.any(catPlot["sourceType"] == 9):
            toPlotList.append((catPlot[xCol], catPlot[yCol], catPlot[zCol], "viridis", "Unknown"))

        # Corner plot of patches showing summary stat in each
        patches = []
        for dataId in sumStats.keys():
            (corners, stat) = sumStats[dataId]
            ra = corners[0][0].asDegrees()
            dec = corners[0][1].asDegrees()
            xy = (ra, dec)
            width = corners[2][0].asDegrees() - ra
            height = corners[2][1].asDegrees() - dec
            patches.append(Rectangle(xy, width, height, alpha=0.3))
            ras = [ra.asDegrees() for (ra, dec) in corners]
            decs = [dec.asDegrees() for (ra, dec) in corners]
            ax.plot(ras + [ras[0]], decs + [decs[0]], "k", lw=0.5)
            cenX = ra + width / 2
            cenY = dec + height / 2
            if dataId != "tract":
                ax.annotate(dataId, (cenX, cenY), color="k", fontsize=7, ha="center", va="center",
                            path_effects=[pathEffects.withStroke(linewidth=2, foreground="w")])

        for (i, (xs, ys, colorVals, cmap, label)) in enumerate(toPlotList):
            med = np.median(colorVals)
            mad = sigmaMad(colorVals)
            vmin = med - 3*mad
            vmax = med + 3*mad
            scatterOut = ax.scatter(xs, ys, c=colorVals, cmap=cmap, s=4.0, vmin=vmin, vmax=vmax)
            cax = fig.add_axes([0.87 + i*0.04, 0.11, 0.04, 0.77])
            plt.colorbar(scatterOut, cax=cax, extend="both")
            colorBarLabel = "{}: {}".format(self.config.axisLabels["z"], label)
            text = cax.text(0.6, 0.5, colorBarLabel, color="k", rotation="vertical", transform=cax.transAxes,
                            ha="center", va="center", fontsize=10)
            text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

            if i == 0 and len(toPlotList) > 1:
                cax.yaxis.set_ticks_position("left")

        ax.set_xlabel(xCol)
        ax.set_ylabel(yCol)

        plt.draw()

        # Add useful information to the plot
        plt.subplots_adjust(wspace=0.0, hspace=0.0, right=0.85)
        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return fig
