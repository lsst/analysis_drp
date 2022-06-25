import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from scipy.stats import binned_statistic_2d
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pathEffects
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import CoordColumn, SingleColumnAction
from lsst.skymap import BaseSkyMap

from .calcFunctors import MagDiff
from .dataSelectors import FlagSelector, SnSelector, StarIdentifier, CoaddPlotFlagSelector
from .plotUtils import generateSummaryStats, parsePlotInfo, addPlotInfo, mkColormap, extremaSort

import pandas as pd

matplotlib.use("Agg")

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
        default={"sourceSelector": StarIdentifier},
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": FlagSelector,
                 "plotFlagSelector": CoaddPlotFlagSelector,
                 "catSnSelector": SnSelector},
    )

    statisticSelectorActions = ConfigurableActionStructField(
        doc="Selectors to use to decide which points to use for calculating statistics.",
        default={"statSelector": SnSelector},
    )

    fixAroundZero = pexConfig.Field(
        doc="Fix the center of the colorscale to be zero.",
        default=False,
        dtype=bool,
    )

    plotOutlines = pexConfig.Field(
        doc="Plot the outlines of the ccds/patches?",
        default=True,
        dtype=bool,
    )

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.xAction.column = "coord_ra"
        self.axisActions.xAction.inRadians = False
        self.axisActions.yAction.column = "coord_dec"
        self.axisActions.yAction.inRadians = False
        self.axisActions.zAction = MagDiff
        self.axisActions.zAction.col1 = "i_ap12Flux"
        self.axisActions.zAction.col2 = "i_psfFlux"
        self.selectorActions.flagSelector.selectWhenFalse = ["i_ap12Flux_flag"]
        self.selectorActions.plotFlagSelector.bands = ["i"]
        self.axisLabels = {"x": "R.A. (Degrees)", "y": "Dec. (Degrees)",
                           "z": "{} - {} (mmag)".format(self.axisActions.zAction.col1.removesuffix("Flux"),
                                                        self.axisActions.zAction.col2.removesuffix("Flux"))}


class SkyPlotTask(pipeBase.PipelineTask):

    ConfigClass = SkyPlotTaskConfig
    _DefaultName = "skyPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        columnNames = set(["patch"])
        bands = []
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.statisticSelectorActions, self.config.sourceSelectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)
                    band = col.split("_")[0]
                    if band not in ["coord", "extend", "detect", "xy", "merge", "sky"]:
                        bands.append(band)

        bands = set(bands)
        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs['catPlot'] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        inputs["plotName"] = localConnections.skyPlot.name
        inputs["bands"] = bands
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, skymap, tableName, bands, plotName):
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

        # Get the S/N cut used (if any)
        if hasattr(self.config.selectorActions, "catSnSelector"):
            SN = self.config.selectorActions.catSnSelector.threshold
            SNFlux = self.config.selectorActions.catSnSelector.fluxType
        else:
            SN = "N/A"
            SNFlux = "N/A"

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, plotName, SN, SNFlux)
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
        self.log.info("Plotting %s: the values of %s on a sky plot.",
                      self.config.connections.plotName, self.config.axisLabels["z"])

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)

        # Make divergent colormaps for stars, galaxes and all the points
        blueGreen = mkColormap(["midnightblue", "lightcyan", "darkgreen"])
        redPurple = mkColormap(["indigo", "lemonchiffon", "firebrick"])
        orangeBlue = mkColormap(["darkOrange", "thistle", "midnightblue"])

        # Need to separate stars and galaxies
        stars = (catPlot["sourceType"] == 1)
        galaxies = (catPlot["sourceType"] == 2)

        xCol = self.config.axisLabels["x"]
        yCol = self.config.axisLabels["y"]
        zCol = self.config.axisLabels["z"]

        # For galaxies
        colorValsGalaxies = catPlot.loc[galaxies, zCol].values
        ids = extremaSort(colorValsGalaxies)
        xsGalaxies = catPlot.loc[galaxies, xCol].values[ids]
        ysGalaxies = catPlot.loc[galaxies, yCol].values[ids]
        colorValsGalaxies = colorValsGalaxies[ids]

        # For stars
        colorValsStars = catPlot.loc[stars, zCol].values
        ids = extremaSort(colorValsStars)
        xsStars = catPlot.loc[stars, xCol].values[ids]
        ysStars = catPlot.loc[stars, yCol].values[ids]
        colorValsStars = colorValsStars[ids]

        # Calculate some statistics
        snForStats = self.config.statisticSelectorActions.statSelector.threshold
        snBands = "".join(self.config.statisticSelectorActions.statSelector.bands)
        snFluxType = self.config.statisticSelectorActions.statSelector.fluxType
        if len(snBands) > 0:
            if "psf" in snFluxType:
                snText = "S/N$_{psf}$" + "_{}>".format(snBands)
            elif "cModel" in snFluxType:
                snText = "S/N$_{cModel}$" + "_{}>".format(snBands)
            else:
                snText = "S/N[{}_{}]>".format(snBands, snFluxType)
        else:
            if "psf" in snFluxType:
                snText = "S/N$_{psf}>$"
            elif "cModel" in snFluxType:
                snText = "S/N_${cModel}>$"
            else:
                snText = "S/N[{}]>".format(snFluxType)
        if np.abs(snForStats) > 1e4:
            snText += "{:0.1g} stats:\n".format(snForStats)
        else:
            snText += "{:0.1f} stats:\n".format(snForStats)
        boxLoc = 0.72
        if np.any(catPlot["sourceType"] == 2):
            statGals = ((catPlot["useForStats"] == 1) & galaxies)
            statGalMed = np.nanmedian(catPlot.loc[statGals, zCol])
            statGalMad = sigmaMad(catPlot.loc[statGals, zCol], nan_policy="omit")

            galStatsText = ("{}".format(snText)
                            + "Median: {:0.2f}\n".format(statGalMed)
                            + r"$\sigma_{MAD}$: " + "{:0.2f}\n".format(statGalMad)
                            + r"n$_{points}$: " + "{}".format(sum(statGals)))
            # Add statistics
            bbox = dict(facecolor="lemonchiffon", alpha=0.5, edgecolor="none")
            # Check if plotting stars and galaxies, if so move the
            # text box so that both can be seen. Needs to be
            # > 2 becuase not being plotted points are assigned 0
            if len(list(set(catPlot["sourceType"].values))) > 2:
                boxLoc -= 0.17
            ax.text(boxLoc, 0.91, galStatsText, transform=fig.transFigure, fontsize=6, bbox=bbox)

        if np.any(catPlot["sourceType"] == 1):

            statStars = ((catPlot["useForStats"] == 1) & stars)
            statStarMed = np.nanmedian(catPlot.loc[statStars, zCol])
            statStarMad = sigmaMad(catPlot.loc[statStars, zCol], nan_policy="omit")

            starStatsText = ("{}".format(snText)
                             + "Median: {:0.2f}\n".format(statStarMed)
                             + r"$\sigma_{MAD}$: " + "{:0.2f}\n".format(statStarMad)
                             + r"n$_{points}$: " + "{}".format(sum(statStars)))
            # Add statistics
            bbox = dict(facecolor="paleturquoise", alpha=0.5, edgecolor="none")
            ax.text(boxLoc, 0.90, starStatsText, transform=fig.transFigure, fontsize=6, bbox=bbox)

        if np.any(catPlot["sourceType"] == 10):

            statAll = (catPlot["useForStats"] == 1)
            statAllMed = np.nanmedian(catPlot.loc[statAll, zCol])
            statAllMad = sigmaMad(catPlot.loc[statAll, zCol], nan_policy="omit")

            allStatsText = ("{}".format(snText)
                            + "Median: {:0.2f}\n".format(statAllMed)
                            + r"$\sigma_{MAD}$: " + "{:0.2f}\n".format(statAllMad)
                            + r"n$_{points}$: " + "{}".format(sum(statAll)))
            bbox = dict(facecolor="purple", alpha=0.2, edgecolor="none")
            ax.text(boxLoc, 0.91, allStatsText, transform=fig.transFigure, fontsize=6, bbox=bbox)

        if np.any(catPlot["sourceType"] == 9):

            statAll = (catPlot["useForStats"] == 1)
            statAllMed = np.nanmedian(catPlot.loc[statAll, zCol])
            statAllMad = sigmaMad(catPlot.loc[statAll, zCol], nan_policy="omit")

            allStatsText = ("{}".format(snText)
                            + "Median: {:0.2f}\n".format(statAllMed)
                            + r"$\sigma_{MAD}$: " + "{:0.2f}\n".format(statAllMad)
                            + r"n$_{points}$: " + "{}".format(sum(statAll)))
            bbox = dict(facecolor="green", alpha=0.2, edgecolor="none")
            ax.text(boxLoc, 0.91, allStatsText, transform=fig.transFigure, fontsize=6, bbox=bbox)

        toPlotList = []
        if np.any(catPlot["sourceType"] == 1):
            toPlotList.append((xsStars, ysStars, colorValsStars, blueGreen, "Stars"))
        if np.any(catPlot["sourceType"] == 2):
            toPlotList.append((xsGalaxies, ysGalaxies, colorValsGalaxies, redPurple, "Galaxies"))
        if np.any(catPlot["sourceType"] == 10):
            ids = extremaSort(catPlot[zCol].values)
            toPlotList.append((catPlot[xCol].values[ids], catPlot[yCol].values[ids],
                               catPlot[zCol].values[ids], orangeBlue, "All"))
        if np.any(catPlot["sourceType"] == 9):
            toPlotList.append((catPlot[xCol], catPlot[yCol], catPlot[zCol], "viridis", "Unknown"))

        # Corner plot of patches showing summary stat in each
        if self.config.plotOutlines:
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
                if dataId == "tract":
                    minRa = np.min(ras)
                    minDec = np.min(decs)
                    maxRa = np.max(ras)
                    maxDec = np.max(decs)
                if dataId != "tract":
                    ax.annotate(dataId, (cenX, cenY), color="k", fontsize=5, ha="center", va="center",
                                path_effects=[pathEffects.withStroke(linewidth=2, foreground="w")])

        for (i, (xs, ys, colorVals, cmap, label)) in enumerate(toPlotList):
            if "tract" not in sumStats.keys() or not self.config.plotOutlines:
                minRa = np.min(xs)
                maxRa = np.max(xs)
                minDec = np.min(ys)
                maxDec = np.max(ys)
                # Avoid identical end points which causes problems in binning
                if minRa == maxRa:
                    maxRa += 1e-5  # There is no reason to pick this number in particular
                if minDec == maxDec:
                    maxDec += 1e-5  # There is no reason to pick this number in particular
            med = np.median(colorVals)
            mad = sigmaMad(colorVals)
            vmin = med - 2*mad
            vmax = med + 2*mad
            if self.config.fixAroundZero:
                scaleEnd = np.max([np.abs(vmin), np.abs(vmax)])
                vmin = -1*scaleEnd
                vmax = scaleEnd
            nBins = 45
            xBinEdges = np.linspace(minRa, maxRa, nBins + 1)
            yBinEdges = np.linspace(minDec, maxDec, nBins + 1)
            binnedStats, xEdges, yEdges, binNums = binned_statistic_2d(xs, ys, colorVals, statistic="median",
                                                                       bins=(xBinEdges, yBinEdges))

            if len(xs) > 5000:
                s = 500/(len(xs)**0.5)
                lw = (s**0.5) / 10
                plotOut = ax.imshow(binnedStats.T, cmap=cmap,
                                    extent=[xEdges[0], xEdges[-1], yEdges[-1], yEdges[0]], vmin=vmin,
                                    vmax=vmax)
                # find the most extreme 15% of points, because the list
                # is ordered by the distance from the median this is just
                # the final 15% of points
                extremes = int(np.floor((len(xs)/100))*85)
                ax.scatter(xs[extremes:], ys[extremes:], c=colorVals[extremes:], s=s, cmap=cmap, vmin=vmin,
                           vmax=vmax, edgecolor="white", linewidths=lw)

            else:
                plotOut = ax.scatter(xs, ys, c=colorVals, cmap=cmap, s=7, vmin=vmin, vmax=vmax,
                                     edgecolor="white", linewidths=0.2)

            cax = fig.add_axes([0.87 + i*0.04, 0.11, 0.04, 0.76])
            plt.colorbar(plotOut, cax=cax, extend="both")
            colorBarLabel = "{}: {}".format(self.config.axisLabels["z"], label)
            text = cax.text(0.5, 0.5, colorBarLabel, color="k", rotation="vertical", transform=cax.transAxes,
                            ha="center", va="center", fontsize=10)
            text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
            cax.tick_params(labelsize=7)

            if i == 0 and len(toPlotList) > 1:
                cax.yaxis.set_ticks_position("left")

        ax.set_xlabel(xCol)
        ax.set_ylabel(yCol)
        ax.tick_params(axis="x", labelrotation=25)
        ax.tick_params(labelsize=7)

        ax.set_aspect("equal")
        plt.draw()

        # Find some useful axis limits
        lenXs = [len(xs) for (xs, _, _, _, _) in toPlotList]
        if lenXs != [] and np.max(lenXs) > 1000:
            padRa = (maxRa - minRa)/10
            padDec = (maxDec - minDec)/10
            ax.set_xlim(maxRa + padRa, minRa - padRa)
            ax.set_ylim(minDec - padDec, maxDec + padDec)
        else:
            ax.invert_xaxis()

        # Add useful information to the plot
        plt.subplots_adjust(wspace=0.0, hspace=0.0, right=0.85, top=0.87)
        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return fig
