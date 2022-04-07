# This file is part of analysis_drp.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import MagColumnNanoJansky, SingleColumnAction
from lsst.skymap import BaseSkyMap

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from . import dataSelectors as dataSelectors
from .plotUtils import generateSummaryStats, parsePlotInfo, addPlotInfo, mkColormap


class ScatterPlotWithTwoHistsTaskConnections(pipeBase.PipelineTaskConnections,
                                             dimensions=("tract", "skymap"),
                                             defaultTemplates={"inputCoaddName": "deep",
                                                               "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="objectTable_tract",
                                             dimensions=("tract", "skymap"),
                                             deferLoad=True)

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
                                            dimensions=("skymap",))

    scatterPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                  storageClass="Plot",
                                                  name="scatterTwoHistPlot_{plotName}",
                                                  dimensions=("tract", "skymap"))


class ScatterPlotWithTwoHistsTaskConfig(pipeBase.PipelineTaskConfig,
                                        pipelineConnections=ScatterPlotWithTwoHistsTaskConnections):

    axisActions = ConfigurableActionStructField(
        doc="The actions to use to calculate the values used on each axis. The defaults for the"
            "column names xAction and magAction are set to iCModelFlux.",
        default={"xAction": MagColumnNanoJansky, "yAction": SingleColumnAction,
                 "magAction": MagColumnNanoJansky},
    )

    axisLabels = pexConfig.DictField(
        doc="Name of the dataframe columns to plot, will be used as the axis label: {'x':, 'y':, 'mag':}"
            "The mag column is used to decide which points to include in the printed statistics.",
        keytype=str,
        itemtype=str
    )

    def get_requirements(self):
        """Return inputs required for a Task to run with this config.

        Returns
        -------
        bands : `set`
            The required bands.
        columns : `set`
            The required column names.
        """
        columnNames = {"patch"}
        bands = set()
        for actionStruct in [self.axisActions,
                             self.selectorActions,
                             self.highSnStatisticSelectorActions,
                             self.lowSnStatisticSelectorActions,
                             self.sourceSelectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    if col is not None:
                        columnNames.add(col)
                        column_split = col.split("_")
                        # If there's no underscore, it has no band prefix
                        if len(column_split) > 1:
                            band = column_split[0]
                            if band not in self.nonBandColumnPrefixes:
                                bands.add(band)
        return bands, columnNames

    nonBandColumnPrefixes = pexConfig.ListField(
        doc="Column prefixes that are not bands and which should not be added to the set of bands",
        dtype=str,
        default=["coord", "extend", "detect", "xy", "merge"],
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": dataSelectors.CoaddPlotFlagSelector},
    )

    highSnStatisticSelectorActions = ConfigurableActionStructField(
        doc="Selectors to use to decide which points to use for calculating the high SN statistics.",
        default={"statSelector": dataSelectors.SnSelector},
    )

    lowSnStatisticSelectorActions = ConfigurableActionStructField(
        doc="Selectors to use to decide which points to use for calculating the low SN statistics.",
        default={"statSelector": dataSelectors.SnSelector},
    )

    sourceSelectorActions = ConfigurableActionStructField(
        doc="What types of sources to use.",
        default={"sourceSelector": dataSelectors.StarIdentifier},
    )

    nBins = pexConfig.Field(
        doc="Number of bins to put on the x axis.",
        default=40.0,
        dtype=float,
    )

    plot2DHist = pexConfig.Field(
        doc="Plot a 2D histogram in the densist area of points on the scatter plot."
            "Doesn't look great if plotting mulitple datasets on top of each other.",
        default=True,
        dtype=bool,
    )

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.magAction.column = "i_cModelFlux"
        self.axisActions.xAction.column = "i_cModelFlux"
        self.highSnStatisticSelectorActions.statSelector.threshold = 2700
        self.lowSnStatisticSelectorActions.statSelector.threshold = 500


class ScatterPlotWithTwoHistsTask(pipeBase.PipelineTask):

    ConfigClass = ScatterPlotWithTwoHistsTaskConfig
    _DefaultName = "scatterPlotWithTwoHistsTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        bands, columnNames = self.config.get_requirements()
        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs['catPlot'] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        inputs["plotName"] = localConnections.scatterPlot.name
        inputs["bands"] = bands
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, skymap, tableName, bands, plotName):
        """Prep the catalogue and then make a scatterPlot of the given column.

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
            scatterPlot : `matplotlib.figure.Figure`
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
        in each patch for the column in self.zColName.

        `scatterPlotWithTwoHists` which makes a scatter plot of the points with
        a histogram of each axis.
        """

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask]

        columns = {self.config.axisLabels["x"]: self.config.axisActions.xAction(catPlot),
                   self.config.axisLabels["y"]: self.config.axisActions.yAction(catPlot),
                   self.config.axisLabels["mag"]: self.config.axisActions.magAction(catPlot),
                   "patch": catPlot["patch"]}
        for actionStruct in [self.config.highSnStatisticSelectorActions,
                             self.config.lowSnStatisticSelectorActions,
                             self.config.sourceSelectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    columns.update({col: catPlot[col]})

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
        useForStats = np.zeros(len(plotDf))
        lowSnMask = np.ones(len(plotDf), dtype=bool)
        for selector in self.config.lowSnStatisticSelectorActions:
            lowSnMask &= selector(plotDf)
        useForStats[lowSnMask] = 2

        highSnMask = np.ones(len(plotDf), dtype=bool)
        for selector in self.config.highSnStatisticSelectorActions:
            highSnMask &= selector(plotDf)
        useForStats[highSnMask] = 1
        plotDf.loc[:, "useForStats"] = useForStats

        # Get the S/N cut used
        try:
            SN = self.config.selectorActions.SnSelector.threshold
        except AttributeError:
            SN = "N/A"

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, plotName, SN)
        # Calculate the corners of the patches and some associated stats
        sumStats = {} if skymap is None else generateSummaryStats(
            plotDf, self.config.axisLabels["y"], skymap, plotInfo)
        # Make the plot
        try:
            fig = self.scatterPlotWithTwoHists(plotDf, plotInfo, sumStats)
        except Exception:
            # This is a workaround until scatterPlotWithTwoHists works properly
            fig = plt.figure(dpi=300)

        return pipeBase.Struct(scatterPlot=fig)

    def scatterPlotWithTwoHists(self, catPlot, plotInfo, sumStats, yLims=False, xLims=False):
        """Makes a generic plot with a 2D histogram and collapsed histograms of
        each axis.

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
            and dec of the corners of the patch, along with a summary
            statistic for each patch.
        yLims : `Bool` or `tuple`, optional
            The y axis limits to use for the plot.  If `False`, they are
            calculated from the data.  If being given a tuple of
            (yMin, yMax).
        xLims : `Bool` or `tuple`, optional
            The x axis limits to use for the plot.  If `False`, they are
            calculated from the data.
            If being given a tuple of (xMin, xMax).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        Uses the axisLabels config options `x` and `y` and the axisAction
        config options `xAction` and `yAction` to plot a scatter
        plot of the values against each other. A histogram of the points
        collapsed onto each axis is also plotted. A summary panel showing the
        median of the y value in each patch is shown in the upper right corner
        of the resultant plot. The code uses the selectorActions to decide
        which points to plot and the statisticSelector actions to determine
        which points to use for the printed statistics.
        """
        self.log.info("Plotting {}: the values of {} on a scatter plot.".format(
                      self.config.connections.plotName, self.config.axisLabels["y"]))

        fig = plt.figure(dpi=300)
        gs = gridspec.GridSpec(4, 4)

        newBlues = mkColormap(["paleturquoise", "midnightBlue"])
        newReds = mkColormap(["lemonchiffon", "firebrick"])

        # Need to separate stars and galaxies
        stars = (catPlot["sourceType"] == 1)
        galaxies = (catPlot["sourceType"] == 2)

        xCol = self.config.axisLabels["x"]
        yCol = self.config.axisLabels["y"]
        magCol = self.config.axisLabels["mag"]

        # For galaxies
        xsGalaxies = catPlot.loc[galaxies, xCol]
        ysGalaxies = catPlot.loc[galaxies, yCol]

        # For stars
        xsStars = catPlot.loc[stars, xCol]
        ysStars = catPlot.loc[stars, yCol]

        highStats = {}
        highMags = {}
        lowStats = {}
        lowMags = {}

        # sourceTypes: 1 - stars, 2 - galaxies, 9 - unknowns
        # 10 - all
        sourceTypeList = [1, 2, 9, 10]
        sourceTypeMapper = {"stars": 1, "galaxies": 2, "unknowns": 9, "all": 10}
        # Calculate some statistics
        for sourceType in sourceTypeList:
            if np.any(catPlot["sourceType"] == sourceType):
                sources = (catPlot["sourceType"] == sourceType)
                highSn = ((catPlot["useForStats"] == 1) & sources)
                highSnMed = np.nanmedian(catPlot.loc[highSn, yCol])
                highSnMad = sigmaMad(catPlot.loc[highSn, yCol], nan_policy="omit")

                lowSn = (((catPlot["useForStats"] == 1) | (catPlot["useForStats"] == 2)) & sources)
                lowSnMed = np.nanmedian(catPlot.loc[lowSn, yCol])
                lowSnMad = sigmaMad(catPlot.loc[lowSn, yCol], nan_policy="omit")

                highStatsStr = ("Median: {:0.2f}    ".format(highSnMed)
                                + r"$\sigma_{MAD}$: " + "{:0.2f}".format(highSnMad))
                highStats[sourceType] = highStatsStr

                lowStatsStr = ("Median: {:0.2f}    ".format(lowSnMed)
                               + r"$\sigma_{MAD}$: " + "{:0.2f}".format(lowSnMad))
                lowStats[sourceType] = lowStatsStr

                if np.sum(highSn) > 0:
                    highMags[sourceType] = f"{np.nanmax(catPlot.loc[highSn, magCol]):.2f}"
                else:
                    highMags[sourceType] = "-"
                if np.sum(lowSn) > 0.0:
                    lowMags[sourceType] = f"{np.nanmax(catPlot.loc[lowSn, magCol]):.2f}"
                else:
                    lowMags[sourceType] = "-"

        # Main scatter plot
        ax = fig.add_subplot(gs[1:, :-1])
        binThresh = 5

        yBinsOut = []
        linesForLegend = []

        if (np.any(catPlot["sourceType"] == sourceTypeMapper["stars"])
                and not np.any(catPlot["sourceType"] == sourceTypeMapper["galaxies"])):
            toPlotList = [(xsStars.values, ysStars.values, "midnightblue", newBlues,
                           sourceTypeMapper["stars"])]
        elif (np.any(catPlot["sourceType"] == sourceTypeMapper["galaxies"])
                and not np.any(catPlot["sourceType"] == sourceTypeMapper["stars"])):
            toPlotList = [(xsGalaxies.values, ysGalaxies.values, "firebrick", newReds,
                           sourceTypeMapper["galaxies"])]
        elif (np.any(catPlot["sourceType"] == sourceTypeMapper["stars"])
                and np.any(catPlot["sourceType"] == sourceTypeMapper["galaxies"])):
            toPlotList = [(xsGalaxies.values, ysGalaxies.values, "firebrick", newReds,
                           sourceTypeMapper["galaxies"]),
                          (xsStars.values, ysStars.values, "midnightblue", newBlues,
                           sourceTypeMapper["stars"])]
        elif np.any(catPlot["sourceType"] == sourceTypeMapper["unknowns"]):
            unknowns = (catPlot["sourceType"] == sourceTypeMapper["unknowns"])
            toPlotList = [(catPlot.loc[unknowns, xCol].values, catPlot.loc[unknowns, yCol].values,
                           "green", None, sourceTypeMapper["unknowns"])]
        elif np.any(catPlot["sourceType"] == sourceTypeMapper["all"]):
            toPlotList = [(catPlot[xCol].values, catPlot[yCol].values, "purple", None,
                           sourceTypeMapper["all"])]
        else:
            toPlotList = []

        for (j, (xs, ys, color, cmap, sourceType)) in enumerate(toPlotList):
            if len(xs) < 2:
                medLine, = ax.plot(xs, np.nanmedian(ys), color,
                                   label="Median: {:0.2f}".format(np.nanmedian(ys)), lw=0.8)
                linesForLegend.append(medLine)
                sigMads = np.array([sigmaMad(ys, nan_policy="omit")]*len(xs))
                sigMadLine, = ax.plot(xs, np.nanmedian(ys) + 1.0*sigMads, color, alpha=0.8, lw=0.8,
                                      label=r"$\sigma_{MAD}$: " + "{:0.2f}".format(sigMads[0]))
                ax.plot(xs, np.nanmedian(ys) - 1.0*sigMads, color, alpha=0.8)
                linesForLegend.append(sigMadLine)
                continue

            [xs1, xs25, xs50, xs75, xs95, xs97] = np.nanpercentile(xs, [1, 25, 50, 75, 95, 97])
            xScale = (xs97 - xs1)/20.0  # This is ~5% of the data range

            # 40 was used as the default number of bins because it looked good
            xEdges = np.arange(np.nanmin(xs) - xScale, np.nanmax(xs) + xScale,
                               (np.nanmax(xs) + xScale - (np.nanmin(xs) - xScale))/self.config.nBins)
            medYs = np.nanmedian(ys)
            sigMadYs = sigmaMad(ys, nan_policy="omit")
            fiveSigmaHigh = medYs + 5.0*sigMadYs
            fiveSigmaLow = medYs - 5.0*sigMadYs
            binSize = (fiveSigmaHigh - fiveSigmaLow)/101.0
            yEdges = np.arange(fiveSigmaLow, fiveSigmaHigh, binSize)

            counts, xBins, yBins = np.histogram2d(xs, ys, bins=(xEdges, yEdges))
            yBinsOut.append(yBins)
            countsYs = np.sum(counts, axis=1)

            ids = np.where((countsYs > binThresh))[0]
            xEdgesPlot = xEdges[ids][1:]
            xEdges = xEdges[ids]

            if len(ids) > 1:
                # Create the codes needed to turn the sigmaMad lines
                # into a path to speed up checking which points are
                # inside the area.
                codes = np.ones(len(xEdgesPlot)*2)*Path.LINETO
                codes[0] = Path.MOVETO
                codes[-1] = Path.CLOSEPOLY

                meds = np.zeros(len(xEdgesPlot))
                threeSigMadVerts = np.zeros((len(xEdgesPlot)*2, 2))
                sigMads = np.zeros(len(xEdgesPlot))

                for (i, xEdge) in enumerate(xEdgesPlot):
                    ids = np.where((xs < xEdge) & (xs > xEdges[i]) & (np.isfinite(ys)))[0]
                    med = np.median(ys[ids])
                    sigMad = sigmaMad(ys[ids])
                    meds[i] = med
                    sigMads[i] = sigMad
                    threeSigMadVerts[i, :] = [xEdge, med + 3*sigMad]
                    threeSigMadVerts[-(i + 1), :] = [xEdge, med - 3*sigMad]

                medLine, = ax.plot(xEdgesPlot, meds, color, label="Running Median")
                linesForLegend.append(medLine)

                # Make path to check which points lie within one sigma mad
                threeSigMadPath = Path(threeSigMadVerts, codes)

                # Add lines for the median +/- 3 * sigma MAD
                threeSigMadLine, = ax.plot(xEdgesPlot, threeSigMadVerts[:len(xEdgesPlot), 1], color,
                                           alpha=0.4, label=r"3$\sigma_{MAD}$")
                ax.plot(xEdgesPlot[::-1], threeSigMadVerts[len(xEdgesPlot):, 1], color, alpha=0.4)

                # Add lines for the median +/- 1 * sigma MAD
                sigMadLine, = ax.plot(xEdgesPlot, meds + 1.0*sigMads, color, alpha=0.8,
                                      label=r"$\sigma_{MAD}$")
                linesForLegend.append(sigMadLine)
                ax.plot(xEdgesPlot, meds - 1.0*sigMads, color, alpha=0.8)

                # Add lines for the median +/- 2 * sigma MAD
                twoSigMadLine, = ax.plot(xEdgesPlot, meds + 2.0*sigMads, color, alpha=0.6,
                                         label=r"2$\sigma_{MAD}$")
                linesForLegend.append(twoSigMadLine)
                linesForLegend.append(threeSigMadLine)
                ax.plot(xEdgesPlot, meds - 2.0*sigMads, color, alpha=0.6)

                # Check which points are outside 3 sigma MAD of the median
                # and plot these as points.
                inside = threeSigMadPath.contains_points(np.array([xs, ys]).T)
                points, = ax.plot(xs[~inside], ys[~inside], ".", ms=3, alpha=0.3, mfc=color, mec=color,
                                  zorder=-1)

                # Add some stats text
                xPos = 0.65 - 0.4*j
                bbox = dict(edgecolor=color, linestyle="--", facecolor="none")
                highThresh = self.config.highSnStatisticSelectorActions.statSelector.threshold
                statText = f"S/N > {highThresh} Stats ({magCol} < {highMags[sourceType]})\n"
                statText += highStats[sourceType]
                fig.text(xPos, 0.087, statText, bbox=bbox, transform=fig.transFigure, fontsize=6)
                if highMags[sourceType] != "-":
                    ax.axvline(float(highMags[sourceType]), color=color, ls="--")

                bbox = dict(edgecolor=color, linestyle=":", facecolor="none")
                lowThresh = self.config.lowSnStatisticSelectorActions.statSelector.threshold
                statText = f"S/N > {lowThresh} Stats ({magCol} < {lowMags[sourceType]})\n"
                statText += lowStats[sourceType]
                fig.text(xPos, 0.017, statText, bbox=bbox, transform=fig.transFigure, fontsize=6)
                if lowMags[sourceType] != "-":
                    ax.axvline(float(lowMags[sourceType]), color=color, ls=":")

                if self.config.plot2DHist:
                    histIm = ax.hexbin(xs[inside], ys[inside], gridsize=75, cmap=cmap, mincnt=1, zorder=-2)

            else:
                points, = ax.plot(xs, ys, ".", ms=5, alpha=0.3, mfc=color, mec=color, zorder=-1)
                meds = np.array([np.nanmedian(ys)]*len(xs))
                medLine, = ax.plot(xs, meds, color, label=f"Median: {np.nanmedian(ys):0.2f}", lw=0.8)
                linesForLegend.append(medLine)
                sigMads = np.array([sigmaMad(ys, nan_policy="omit")]*len(xs))
                sigMadLine, = ax.plot(xs, meds + 1.0*sigMads, color, alpha=0.8, lw=0.8,
                                      label=r"$\sigma_{MAD}$: " + f"{sigMads[0]:0.2f}")
                ax.plot(xs, meds - 1.0*sigMads, color, alpha=0.8)
                linesForLegend.append(sigMadLine)
                histIm = None

        # Set the scatter plot limits
        if len(ysStars) > 0:
            plotMed = np.nanmedian(ysStars)
        else:
            plotMed = np.nanmedian(ysGalaxies)
        if yLims:
            ax.set_ylim(yLims[0], yLims[1])
        else:
            numSig = 4
            yLimMin = plotMed - numSig*sigMadYs
            yLimMax = plotMed + numSig*sigMadYs
            while (yLimMax < np.max(meds) or yLimMin > np.min(meds)) and numSig < 10:
                numSig += 1

            numSig += 1
            yLimMin = plotMed - numSig*sigMadYs
            yLimMax = plotMed + numSig*sigMadYs
            ax.set_ylim(yLimMin, yLimMax)

        if xLims:
            ax.set_xlim(xLims[0], xLims[1])
        else:
            ax.set_xlim(xs1 - xScale, xs97 + xScale)

        # Add a line legend
        ax.legend(handles=linesForLegend, ncol=4, fontsize=6, loc="upper left", framealpha=0.9,
                  edgecolor="k", borderpad=0.4, handlelength=1)

        # Add axes labels
        ax.set_ylabel(yCol, fontsize=10, labelpad=10)
        ax.set_xlabel(xCol, fontsize=10, labelpad=2)

        # Top histogram
        topHist = plt.gcf().add_subplot(gs[0, :-1], sharex=ax)
        topHist.hist(catPlot[xCol].values, bins=100, color="grey", alpha=0.3, log=True,
                     label=f"All ({len(catPlot)})")
        if np.any(catPlot["sourceType"] == 2):
            topHist.hist(xsGalaxies, bins=100, color="firebrick", histtype="step", log=True,
                         label=f"Galaxies ({len(np.where(galaxies)[0])})")
        if np.any(catPlot["sourceType"] == 1):
            topHist.hist(xsStars, bins=100, color="midnightblue", histtype="step", log=True,
                         label=f"Stars ({len(np.where(stars)[0])})")
        topHist.axes.get_xaxis().set_visible(False)
        topHist.set_ylabel("Number", fontsize=8)
        topHist.legend(fontsize=6, framealpha=0.9, borderpad=0.4, loc="lower left", ncol=3, edgecolor="k")

        # Side histogram
        sideHist = plt.gcf().add_subplot(gs[1:, -1], sharey=ax)
        finiteObjs = np.isfinite(catPlot[yCol].values)
        bins = np.linspace(yLimMin, yLimMax)
        sideHist.hist(catPlot[yCol].values[finiteObjs], bins=bins, color="grey", alpha=0.3,
                      orientation="horizontal", log=True)
        if np.any(catPlot["sourceType"] == sourceTypeMapper["galaxies"]):
            sideHist.hist(ysGalaxies[np.isfinite(ysGalaxies)], bins=bins, color="firebrick", histtype="step",
                          orientation="horizontal", log=True)
            if highMags[sourceTypeMapper["galaxies"]] != "-":
                sideHist.hist(ysGalaxies[np.isfinite(ysGalaxies) & (xsGalaxies < float(highMags[2]))],
                              bins=bins, color="firebrick", histtype="step", orientation="horizontal",
                              log=True, ls="--")
            if lowMags[sourceTypeMapper["galaxies"]] != "-":
                sideHist.hist(ysGalaxies[np.isfinite(ysGalaxies) & (xsGalaxies < float(lowMags[2]))],
                              bins=bins, color="firebrick", histtype="step", orientation="horizontal",
                              log=True, ls=":")

        if np.any(catPlot["sourceType"] == sourceTypeMapper["stars"]):
            sideHist.hist(ysStars[np.isfinite(ysStars)], bins=bins, color="midnightblue", histtype="step",
                          orientation="horizontal", log=True)
            if highMags[sourceTypeMapper["stars"]] != "-":
                sideHist.hist(ysStars[np.isfinite(ysStars) & (xsStars < float(highMags[1]))], bins=bins,
                              color="midnightblue", histtype="step", orientation="horizontal", log=True,
                              ls="--")
            if lowMags[sourceTypeMapper["stars"]] != "-":
                sideHist.hist(ysStars[np.isfinite(ysStars) & (xsStars < float(lowMags[1]))], bins=bins,
                              color="midnightblue", histtype="step", orientation="horizontal", log=True,
                              ls=":")

        sideHist.axes.get_yaxis().set_visible(False)
        sideHist.set_xlabel("Number", fontsize=8)
        if self.config.plot2DHist and histIm is not None:
            divider = make_axes_locatable(sideHist)
            cax = divider.append_axes("right", size="8%", pad=0)
            plt.colorbar(histIm, cax=cax, orientation="vertical", label="Number of Points Per Bin")

        # Corner plot of patches showing summary stat in each
        axCorner = plt.gcf().add_subplot(gs[0, -1])
        axCorner.yaxis.tick_right()
        axCorner.yaxis.set_label_position("right")
        axCorner.xaxis.tick_top()
        axCorner.xaxis.set_label_position("top")
        axCorner.set_aspect("equal")

        patches = []
        colors = []
        for dataId in sumStats.keys():
            (corners, stat) = sumStats[dataId]
            ra = corners[0][0].asDegrees()
            dec = corners[0][1].asDegrees()
            xy = (ra, dec)
            width = corners[2][0].asDegrees() - ra
            height = corners[2][1].asDegrees() - dec
            patches.append(Rectangle(xy, width, height))
            colors.append(stat)
            ras = [ra.asDegrees() for (ra, dec) in corners]
            decs = [dec.asDegrees() for (ra, dec) in corners]
            axCorner.plot(ras + [ras[0]], decs + [decs[0]], "k", lw=0.5)
            cenX = ra + width / 2
            cenY = dec + height / 2
            if dataId != "tract":
                axCorner.annotate(dataId, (cenX, cenY), color="k", fontsize=4, ha="center", va="center")

        cmapUse = plt.cm.coolwarm
        # Set the bad color to transparent and make a masked array
        cmapUse.set_bad(color="none")
        colors = np.ma.array(colors, mask=np.isnan(colors))
        collection = PatchCollection(patches, cmap=cmapUse)
        collection.set_array(colors)
        axCorner.add_collection(collection)

        axCorner.set_xlabel("R.A. (deg)", fontsize=7)
        axCorner.set_ylabel("Dec. (deg)", fontsize=7)
        axCorner.tick_params(axis="both", labelsize=6, length=0, pad=1.5)
        axCorner.invert_xaxis()

        # Add a colorbar
        pos = axCorner.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 + 0.23, pos.x1 - pos.x0, 0.025])
        plt.colorbar(collection, cax=cax, orientation="horizontal")
        cax.text(0.5, 0.5, "Median Value", color="k", transform=cax.transAxes, rotation="horizontal",
                 horizontalalignment="center", verticalalignment="center", fontsize=6)
        cax.tick_params(axis="x", labelsize=6, labeltop=True, labelbottom=False, bottom=False, top=True,
                        pad=0.5, length=2)

        plt.draw()
        plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.22, left=0.21)
        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return fig
