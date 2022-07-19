import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation as sigmaMad
import scipy.stats
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pathEffects

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import MagColumnNanoJansky

from .calcFunctors import ExtinctionCorrectedMagDiff
from .dataSelectors import StarIdentifier, CoaddPlotFlagSelector, SnSelector
from .plotUtils import parsePlotInfo, addPlotInfo, stellarLocusFit, perpDistance, mkColormap

matplotlib.use("Agg")


class ColorColorFitPlotConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("tract", "skymap"),
                                   defaultTemplates={"plotName": "wFit"}):

    catPlot = pipeBase.connectionTypes.Input(
        doc="The tract wide catalog to make plots from.",
        storageClass="DataFrame",
        name="objectTable_tract",
        dimensions=("tract", "skymap"),
        deferLoad=True)

    colorColorFitPlot = pipeBase.connectionTypes.Output(
        doc="A color-color plot with a fit to the stellar locus.",
        storageClass="Plot",
        name="colorColorFitPlot_{plotName}",
        dimensions=("tract", "skymap"))


class ColorColorFitPlotConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=ColorColorFitPlotConnections):

    axisActions = ConfigurableActionStructField(
        doc="The actions to use to calculate the values on each axis.",
        default={"xAction": ExtinctionCorrectedMagDiff, "yAction": ExtinctionCorrectedMagDiff,
                 "magAction": MagColumnNanoJansky},
    )

    axisLabels = pexConfig.DictField(
        doc="Axis labels for the plot.",
        keytype=str,
        itemtype=str,
        default={"x": "g - r PSF (mag)",
                 "y": "r - i PSF (mag)",
                 "mag": "r PSF (mag)"},
    )

    sourceSelectorActions = ConfigurableActionStructField(
        doc="What types of sources to use.",
        default={"sourceSelector": StarIdentifier},
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": CoaddPlotFlagSelector,
                 "catSnSelector": SnSelector},
    )

    stellarLocusFitDict = pexConfig.DictField(
        doc="The parameters to use for the stellar locus fit. The default parameters are examples and are "
            "not useful for any of the fits. The dict needs to contain xMin/xMax/yMin/yMax which are the "
            "limits of the initial box for fitting the stellar locus, mHW and bHW are the initial "
            "intercept and gradient for the fitting.",
        keytype=str,
        itemtype=float,
        default={"xMin": 0.1, "xMax": 0.2, "yMin": 0.1, "yMax": 0.2, "mHW": 0.5, "bHW": 0.0}
    )

    bands = pexConfig.DictField(
        doc="Names of the bands to use for the colors.  Plots are of band2 - band3 vs. band2 - band1",
        keytype=str,
        itemtype=str,
        default={"band1": "g", "band2": "r", "band3": "i"}
    )

    fluxTypeForColor = pexConfig.Field(
        doc="Flavor of flux measurement to use for colors",
        default="psfFlux",
        dtype=str,
    )

    xLims = pexConfig.ListField(
        doc="Minimum and maximum x-axis limit to force (provided as a list of [xMin, xMax]). "
        "If `None`, limits will be computed and set based on the data.",
        dtype=float,
        default=None,
        optional=True,
    )

    yLims = pexConfig.ListField(
        doc="Minimum and maximum y-axis limit to force (provided as a list of [yMin, yMax]). "
        "If `None`, limits will be computed and set based on the data.",
        dtype=float,
        default=None,
        optional=True,
    )

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.xAction.magDiff.returnMillimags = False
        self.axisActions.yAction.magDiff.returnMillimags = False
        self.axisActions.magAction.column = "r_psfFlux"
        self.stellarLocusFitDict = {"xMin": 0.28, "xMax": 1.0, "yMin": 0.02, "yMax": 0.48,
                                    "mHW": 0.52, "bHW": -0.08}
        self.xLims = (-0.7, 2.3)
        self.yLims = (-0.7, 2.6)
        self.setConfigDependencies()

    def setConfigDependencies(self):
        # The following config settings are conditional on other configs.
        # Set them based on the inter-dependencies here to ensure they are
        # in sync.  This can (and should) be called in the pipeline definition
        # if any of the inter-dependent configs are changed (e.g. self.bands
        # or self.fluxTypeForColor here.  See plot_wFit_CModel in
        # stellarLocusPlots.yaml for an example use case.
        band1 = self.bands["band1"]
        band2 = self.bands["band2"]
        band3 = self.bands["band3"]
        fluxTypeStr = self.fluxTypeForColor.removesuffix("Flux")
        fluxFlagStr = fluxTypeStr if "cModel" in fluxTypeStr else self.fluxTypeForColor
        self.selectorActions.flagSelector.selectWhenFalse = [band1 + "_" + fluxFlagStr + "_flag",
                                                             band2 + "_" + fluxFlagStr + "_flag",
                                                             band3 + "_" + fluxFlagStr + "_flag"]
        self.axisActions.xAction.magDiff.col1 = band1 + "_" + self.fluxTypeForColor
        self.axisActions.xAction.magDiff.col2 = band2 + "_" + self.fluxTypeForColor
        self.axisActions.yAction.magDiff.col1 = band2 + "_" + self.fluxTypeForColor
        self.axisActions.yAction.magDiff.col2 = band3 + "_" + self.fluxTypeForColor
        self.selectorActions.flagSelector.bands = (band1, band2, band3)
        self.selectorActions.catSnSelector.bands = (band1, band2, band3)
        self.selectorActions.catSnSelector.fluxType = self.fluxTypeForColor
        self.axisLabels = {"x": band1 + " - " + band2 + " " + fluxTypeStr + " (mag)",
                           "y": band2 + " - " + band3 + " " + fluxTypeStr + " (mag)",
                           "mag": self.axisActions.magAction.column.removesuffix("Flux") + " (mag)"}


class ColorColorFitPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorFitPlotConfig
    _DefaultName = "colorColorFitPlot"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        columnNames = set()
        bands = []
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.sourceSelectorActions]:
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
        inputs["plotName"] = localConnections.colorColorFitPlot.name
        inputs["bands"] = bands
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, tableName, bands, plotName):

        mask = np.ones(len(catPlot), dtype=bool)
        for actionStruct in [self.config.selectorActions, self.config.sourceSelectorActions]:
            for selector in actionStruct:
                mask &= selector(catPlot)
        catPlot = catPlot[mask]

        columns = {self.config.axisLabels["x"]: self.config.axisActions.xAction(catPlot),
                   self.config.axisLabels["y"]: self.config.axisActions.yAction(catPlot),
                   self.config.axisLabels["mag"]: self.config.axisActions.magAction(catPlot)}

        # Get the S/N cut used (if any)
        if hasattr(self.config.selectorActions, "catSnSelector"):
            for col in self.config.selectorActions.catSnSelector.columns:
                columns[col] = catPlot[col]
            SN = self.config.selectorActions.catSnSelector.threshold
            SNFlux = self.config.selectorActions.catSnSelector.fluxType
        else:
            SN = "N/A"
            SNFlux = "N/A"

        plotDf = pd.DataFrame(columns)

        xs = plotDf[self.config.axisLabels["x"]].values
        ys = plotDf[self.config.axisLabels["y"]].values

        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, plotName, SN, SNFlux)
        if len(plotDf) == 0:
            fig = plt.Figure()
            noDataText = ("No data to plot after selectors applied\n(do you have all three of "
                          "the bands required: {}?)".format(bands))
            fig.text(0.5, 0.5, noDataText, ha="center", va="center")
            fig = addPlotInfo(fig, plotInfo)
        else:
            fitParams = stellarLocusFit(xs, ys, self.config.stellarLocusFitDict)
            fig = self.colorColorFitPlot(plotDf, plotInfo, fitParams)

        return pipeBase.Struct(colorColorFitPlot=fig)

    def colorColorFitPlot(self, catPlot, plotInfo, fitParams):
        """Make stellar locus plots using pre fitted values.

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
                ``"bands"``
                    The bands used for the plot (`str`).
                ``"SN"``
                    The minimum signal-to-noise threshold (`float` or `str`).
                ``"SNFlux"``
                    The flux used for the signal-to-noise to select on
                    (`float` or `str`).
        fitParams : `dict`
            The parameters of the fit to the stellar locus calculated
            elsewhere, they are used to plot the fit line on the
            figure.
                ``"bODR"``
                    The intercept calculated by the orthogonal distance
                    regression fitting.
                ``"mODR"``
                    The gradient calculated by the orthogonal distance
                    regression fitting.
                ``"fitPoints"``
                    A boolean array indicating which points were used in the
                    final ODR fit (`numpy.ndarray` [`bool`]).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        Makes a color-color plot of `self.config.xColName` against
        `self.config.yColName`, these points are color coded by i band
        CModel magnitude. The stellar locus fits calculated from
        the calcStellarLocus task are then overplotted. The axis labels
        are given by `self.config.xLabel` and `self.config.yLabel`.
        The selector given in `self.config.sourceSelectorActions`
        is used for source selection. The distance of the points to
        the fit line is given in a histogram in the second panel.
        """

        self.log.info(("Plotting %s: the values of %s against %s on a color-color plot with the area "
                       "used for calculating the stellar locus fits marked.",
                       self.config.connections.plotName, self.config.axisLabels["x"],
                       self.config.axisLabels["y"]))

        # Define a new colormap
        newBlues = mkColormap(["darkblue", "paleturquoise"])
        newGrays = mkColormap(["lightslategray", "white"])

        # Make a figure with three panels
        fig = plt.figure(dpi=300)
        ax = fig.add_axes([0.12, 0.25, 0.43, 0.60])
        axContour = fig.add_axes([0.65, 0.11, 0.3, 0.31])
        axHist = fig. add_axes([0.65, 0.51, 0.3, 0.31])
        xs = catPlot[self.config.axisLabels["x"]].values
        ys = catPlot[self.config.axisLabels["y"]].values
        mags = catPlot[self.config.axisLabels["mag"]].values

        if len(xs) == 0 or len(ys) == 0:
            return fig

        # Points used for the fit
        fitPoints = fitParams["fitPoints"]

        # Add some useful information to the plot
        SNsUsedDict = {}
        if hasattr(self.config.selectorActions, "catSnSelector"):
            # Compute the effective S/N & mag cuts for the points used in the
            # fit.
            for SNBand in self.config.selectorActions.catSnSelector.bands:
                SNsUsed = (catPlot[SNBand + "_" + plotInfo["SNFlux"]].values[fitPoints]
                           / catPlot[SNBand + "_" + plotInfo["SNFlux"] + "Err"].values[fitPoints])
                minSnUsed = np.nanmin(SNsUsed)
                magsUsed = mags[fitPoints]
                incr = 5.0
                idsUsed = (SNsUsed < minSnUsed + incr)
                while sum(idsUsed) < max(0.005*len(idsUsed), 3):
                    incr += 5.0
                    idsUsed = (SNsUsed < plotInfo["SN"] + incr)
                medMagUsed = np.nanmedian(magsUsed[idsUsed])
                SNsUsedDict[SNBand] = {"minSnUsed": minSnUsed, "medMagUsed": medMagUsed}

        bbox = dict(alpha=0.9, facecolor="white", edgecolor="none")
        infoText = "N Total: {}".format(len(catPlot))
        ax.text(0.04, 0.97, infoText, color="k", transform=ax.transAxes,
                fontsize=7, bbox=bbox, va="top")
        infoText = "N Used in fit: {}".format(sum(fitPoints))
        ax.text(0.04, 0.925, infoText, color="darkblue", transform=ax.transAxes,
                fontsize=7, bbox=bbox, va="top")
        if len(SNsUsedDict) > 0:
            yLoc = 0.9
            for SNBand in self.config.selectorActions.catSnSelector.bands:
                yLoc -= 0.04
                infoText = "{}: S/N".format(SNBand)
                infoText += r"$\geq$" + "{:0.1f} [".format(SNsUsedDict[SNBand]["minSnUsed"])
                infoText += r"$\lesssim$ " + "{:0.1f} mag]".format(SNsUsedDict[SNBand]["medMagUsed"])
                ax.text(0.04, yLoc, infoText, color="C0", transform=ax.transAxes,
                        fontsize=6, va="center")

        # Calculate the density of the points
        xyUsed = np.vstack([xs[fitPoints], ys[fitPoints]])
        xyNotUsed = np.vstack([xs[~fitPoints], ys[~fitPoints]])
        zUsed = scipy.stats.gaussian_kde(xyUsed)(xyUsed)
        zNotUsed = scipy.stats.gaussian_kde(xyNotUsed)(xyNotUsed)

        notUsedScatter = ax.scatter(xs[~fitPoints], ys[~fitPoints], c=zNotUsed, cmap=newGrays,
                                    s=0.3)
        fitScatter = ax.scatter(xs[fitPoints], ys[fitPoints], c=zUsed, cmap=newBlues,
                                label="Used for Fit", s=0.3)

        # Add colorbars
        cbAx = fig.add_axes([0.12, 0.07, 0.43, 0.04])
        plt.colorbar(fitScatter, cax=cbAx, orientation="horizontal")
        cbText = cbAx.text(0.5, 0.5, "Number Density (used in fit)", color="k",
                           rotation="horizontal", transform=cbAx.transAxes,
                           ha="center", va="center", fontsize=7)
        cbText.set_path_effects([pathEffects.Stroke(linewidth=1.5, foreground="w"), pathEffects.Normal()])
        cbAx.set_xticks([np.min(zUsed), np.max(zUsed)], labels=["Less", "More"], fontsize=7)
        cbAxNotUsed = fig.add_axes([0.12, 0.11, 0.43, 0.04])
        plt.colorbar(notUsedScatter, cax=cbAxNotUsed, orientation="horizontal")
        cbText = cbAxNotUsed.text(0.5, 0.5, "Number Density (not used in fit)", color="k",
                                  rotation="horizontal", transform=cbAxNotUsed.transAxes,
                                  ha="center", va="center", fontsize=7)
        cbText.set_path_effects([pathEffects.Stroke(linewidth=1.5, foreground="w"),
                                 pathEffects.Normal()])
        cbAxNotUsed.set_xticks([])

        ax.set_xlabel(self.config.axisLabels["x"])
        ax.set_ylabel(self.config.axisLabels["y"])

        # Set useful axis limits
        if self.config.xLims is not None:
            ax.set_xlim(self.config.xLims[0], self.config.xLims[1])
        else:
            percsX = np.nanpercentile(xs, [0.5, 99.5])
            x5 = (percsX[1] - percsX[0])/5
            ax.set_xlim(percsX[0] - x5, percsX[1] + x5)
        if self.config.yLims is not None:
            ax.set_ylim(self.config.yLims[0], self.config.yLims[1])
        else:
            percsY = np.nanpercentile(ys, [0.5, 99.5])
            y5 = (percsY[1] - percsY[0])/5
            ax.set_ylim(percsY[0] - y5, percsY[1] + y5)

        # Plot the fit lines
        xMin = self.config.stellarLocusFitDict["xMin"]
        xMax = self.config.stellarLocusFitDict["xMax"]
        yMin = self.config.stellarLocusFitDict["yMin"]
        yMax = self.config.stellarLocusFitDict["yMax"]
        mHW = self.config.stellarLocusFitDict["mHW"]
        bHW = self.config.stellarLocusFitDict["bHW"]

        if np.fabs(mHW) > 1:
            ysFitLineHW = np.array([yMin, yMax])
            xsFitLineHW = (ysFitLineHW - bHW)/mHW
            ysFitLine = np.array([yMin, yMax])
            xsFitLine = (ysFitLine - fitParams["bODR"])/fitParams["mODR"]
        else:
            xsFitLineHW = np.array([xMin, xMax])
            ysFitLineHW = mHW*xsFitLineHW + bHW
            xsFitLine = [xMin, xMax]
            ysFitLine = [fitParams["mODR"]*xsFitLine[0] + fitParams["bODR"],
                         fitParams["mODR"]*xsFitLine[1] + fitParams["bODR"]]

        ax.plot(xsFitLineHW, ysFitLineHW, "w", lw=1.5)
        lineHW, = ax.plot(xsFitLineHW, ysFitLineHW, "g", lw=1, ls="--", label="Hardwired")
        ax.plot(xsFitLine, ysFitLine, "w", lw=1.5)
        lineRefit, = ax.plot(xsFitLine, ysFitLine, "k", lw=1, ls="--", label="ODR Fit")
        ax.legend(handles=[lineHW, lineRefit], fontsize=6, loc="lower right")

        # Calculate the distances (in mmag) to that line for the data used in
        # the fit.  Two points are needed to characterise the lines we want to
        # get the distances to.
        p1 = np.array([xsFitLine[0], ysFitLine[0]])
        p2 = np.array([xsFitLine[1], ysFitLine[1]])

        p1HW = np.array([xsFitLineHW[0], ysFitLineHW[0]])
        p2HW = np.array([xsFitLineHW[1], ysFitLineHW[1]])

        distsHW = perpDistance(p1HW, p2HW, zip(xs[fitPoints], ys[fitPoints]))
        dists = perpDistance(p1, p2, zip(xs[fitPoints], ys[fitPoints]))
        maxDist = np.abs(np.nanmax(dists))  # These will be used to set the fit boundary line limits
        minDist = np.abs(np.nanmin(dists))
        # Convert dists units to mmag
        statsUnitScale = 1000.0  # I.e. quote the statistics in mmag
        statsUnitStr = "mmag"
        distsHW = [dist*statsUnitScale for dist in distsHW]
        dists = [dist*statsUnitScale for dist in dists]

        # Now we have the information for the perpendicular line we
        # can use it to calculate the points at the ends of the
        # perpendicular lines that intersect at the box edges
        if np.fabs(mHW) > 1:
            xMid = (yMin - fitParams["bODR"])/fitParams["mODR"]
            xs = np.array([xMid - max(0.2, maxDist), xMid, xMid + max(0.2, minDist)])
            ys = fitParams["mPerp"]*xs + fitParams["bPerpMin"]
        else:
            xs = np.array([xMin - max(0.2, np.fabs(mHW)*maxDist), xMin,
                           xMin + max(0.2, np.fabs(mHW)*minDist)])
            ys = xs*fitParams["mPerp"] + fitParams["bPerpMin"]
        ax.plot(xs, ys, "k--", alpha=0.7, lw=1)

        if np.fabs(mHW) > 1:
            xMid = (yMax - fitParams["bODR"])/fitParams["mODR"]
            xs = np.array([xMid - max(0.2, maxDist), xMid, xMid + max(0.2, minDist)])
            ys = fitParams["mPerp"]*xs + fitParams["bPerpMax"]
        else:
            xs = np.array([xMax - max(0.2, np.fabs(mHW)*maxDist), xMax,
                           xMax + max(0.2, np.fabs(mHW)*minDist)])
            ys = xs*fitParams["mPerp"] + fitParams["bPerpMax"]
        ax.plot(xs, ys, "k--", alpha=0.7, lw=1)

        # Compute statistics for fit
        dists = np.asarray(dists)
        distsHW = np.asarray(distsHW)
        medDists = np.median(dists)
        madDists = sigmaMad(dists, scale="normal")
        meanDists = np.mean(dists)
        rmsDists = np.sqrt(np.mean(np.array(dists)**2))

        # Add a histogram
        axHist.set_ylabel("Number", fontsize=8)
        axHist.set_xlabel("Distance to Line Fit ({})".format(statsUnitStr), fontsize=8)
        axHist.tick_params(labelsize=8)
        nSigToPlot = 3.5
        axHist.set_xlim(meanDists - nSigToPlot*madDists,
                        meanDists + nSigToPlot*madDists)
        lineMedian = axHist.axvline(medDists, color="k", label="Median: {:0.2f} {}".
                                    format(medDists, statsUnitStr))
        lineMad = axHist.axvline(medDists + madDists, color="k", ls="--",
                                 label=r"$\sigma_{MAD}$" + ": {:0.2f} {}"
                                 .format(madDists, statsUnitStr))
        axHist.axvline(medDists - madDists, color="k", ls="--")
        lineRms = axHist.axvline(meanDists + rmsDists, color="C0", ls="--",
                                 label="RMS: {:0.2f} {}".format(rmsDists, statsUnitStr))
        axHist.axvline(meanDists - rmsDists, color="C0", ls="--")

        linesForLegend = [lineMedian, lineMad, lineRms]
        fig.legend(handles=linesForLegend, fontsize=7, loc="lower right", bbox_to_anchor=(0.96, 0.84),
                   bbox_transform=fig.transFigure, ncol=1)

        axHist.hist(dists, bins=200, histtype="step", label="ODR Fit", color="C0")
        axHist.hist(distsHW, bins=200, histtype="step", label="HW", color="C0", alpha=0.5)

        alphas = [1.0, 0.5]
        handles = [Rectangle((0, 0), 1, 1, color="none", ec="C0", alpha=a) for a in alphas]
        labels = ["ODR Fit", "HW"]
        axHist.legend(handles, labels, fontsize=6, loc="upper right")

        # Add a contour plot showing the magnitude dependance
        # of the distance to the fit
        axContour.invert_yaxis()
        axContour.axvline(0.0, color="k", ls="--", zorder=-1)
        percsDists = np.nanpercentile(dists, [4, 96])
        minXs = -1*np.min(np.fabs(percsDists))
        maxXs = np.min(np.fabs(percsDists))
        plotPoints = ((dists < maxXs) & (dists > minXs))
        xs = np.array(dists)[plotPoints]
        ys = mags[fitPoints][plotPoints]
        H, xEdges, yEdges = np.histogram2d(xs, ys, bins=(11, 11))
        xBinWidth = xEdges[1] - xEdges[0]
        yBinWidth = yEdges[1] - yEdges[0]
        axContour.contour(xEdges[:-1] + xBinWidth/2, yEdges[:-1] + yBinWidth/2, H.T, levels=7,
                          cmap=newBlues)
        axContour.set_xlabel("Distance to Line Fit ({})".format(statsUnitStr), fontsize=8)
        axContour.set_ylabel(self.config.axisLabels["mag"], fontsize=8)
        axContour.set_xlim(meanDists - nSigToPlot*madDists,
                           meanDists + nSigToPlot*madDists)
        axContour.tick_params(labelsize=8)

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
