import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pathEffects
import numpy as np
import pandas as pd
import scipy.stats as scipyStats

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import MagColumnNanoJansky
from .calcFunctors import ExtinctionCorrectedMagDiff
from .dataSelectors import FlagSelector, SnSelector, StarIdentifier, GalaxyIdentifier, CoaddPlotFlagSelector
from .plotUtils import parsePlotInfo, addPlotInfo, mkColormap

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
        default={"starIdentifier": StarIdentifier,
                 "galaxyIdentifier": GalaxyIdentifier},
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": CoaddPlotFlagSelector,
                 "extraFlagSelector": FlagSelector,
                 "catSnSelector": SnSelector},
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

    nBins = pexConfig.Field(
        doc="Number of bins for 2d histograms.  Ignored if ``config.contourPlot`` is `False",
        default=40,
        dtype=int,
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

    contourPlot = pexConfig.Field(
        doc="Plot contours by point density instead of points colormapped by mag. "
            "These take a while to make, so should probably not be included by default "
            "in production runs.",
        default=False,
        dtype=bool,
    )

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.zAction.column = "i_psfFlux"
        self.axisActions.xAction.magDiff.returnMillimags = False
        self.axisActions.yAction.magDiff.returnMillimags = False
        self.selectorActions.catSnSelector.fluxType = "psfFlux"
        self.selectorActions.catSnSelector.threshold = 50
        self.setConfigDependencies()

    def setConfigDependencies(self):
        # The following config settings are conditional on other configs.
        # Set them based on the inter-dependencies here to ensure they are
        # in sync.  This can (and should) be called in the pipeline definition
        # if any of the inter-dependent configs are chaned (e.g. self.bands
        # or self.fluxTypeForColor here.  See plot_iz_ri_psf in
        # coaddQAPlots.yaml for an example use case.
        band1 = self.bands["band1"]
        band2 = self.bands["band2"]
        band3 = self.bands["band3"]
        fluxTypeStr = self.fluxTypeForColor.removesuffix("Flux")
        fluxFlagStr = fluxTypeStr if "cModel" in fluxTypeStr else self.fluxTypeForColor
        self.selectorActions.extraFlagSelector.selectWhenFalse = [band1 + "_" + fluxFlagStr + "_flag",
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
                           "z": self.axisActions.zAction.column.removesuffix("Flux") + " (mag)"}


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
            if self.config.contourPlot:
                fig = self.colorColorContourPlot(plotDf, plotInfo)
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

        self.log.info("Plotting %s: %s against %s on a color-color plot.",
                      self.config.connections.plotName, self.config.axisLabels["x"],
                      self.config.axisLabels["y"])

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
        if galPoints:
            galBBox = dict(facecolor="lemonchiffon", alpha=0.5, edgecolor="none")
            fig.text(0.70, 0.9, "Num. Galaxies: {}".format(galaxies.sum()), bbox=galBBox, fontsize=8)
        if starPoints:
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
        if self.config.xLims is not None:
            ax.set_xlim(self.config.xLims[0], self.config.xLims[1])
        else:
            if len(xsStars) > 0:
                starPercsX = np.nanpercentile(xsStars, [1, 99.5])
                pad = (starPercsX[1] - starPercsX[0])/10
                ax.set_xlim(starPercsX[0] - pad, starPercsX[1] + pad)

        if self.config.yLims is not None:
            ax.set_ylim(self.config.yLims[0], self.config.yLims[1])
        else:
            if len(ysStars) > 0:
                starPercsY = np.nanpercentile(ysStars, [1, 99.5])
                pad = (starPercsY[1] - starPercsY[0])/10
                ax.set_ylim(starPercsY[0] - pad, starPercsY[1] + pad)

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig

    def colorColorContourPlot(self, catPlot, plotInfo):
        """Make color-color contour plots for stars and galaxies.

        This creates a two panel contour plot (left: stars, right: galaxies).

        The point density is also computed as a gaussian kde and any point
        with a density below the threshold set in maxDens is plotted as a black
        point.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog of data from which to make plot.
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
        Makes a contour color-color plot of `self.config.axisLabels['y']`
        against `self.config.axisLabels['x']`, the contours are coded by
        point denstiy (computed as a 2D histogram).
        The axis labels are given by `self.config.xLabel`
        and `self.config.yLabel`. The column given in
        `self.config.sourceTypeColName` is used for star galaxy separation.

        These plots take a while to make, so should probably be tier 2
        at the most (i.e. not run by default in DRP campaigns).
        """
        self.log.info("Plotting %s: %s against %s as a contour plot.",
                      self.config.connections.plotName, self.config.axisLabels["x"],
                      self.config.axisLabels["y"])

        minPoints = 5  # Minimum number of points for lowest contour
        nLevel = 7  # Number of contour levels to plot
        maxDens = 0.03  # Plot points for densities lower than this threshold

        # Define a new colormap
        newBlues = mkColormap(["paleturquoise", "darkblue"])
        newReds = mkColormap(["lemonchiffon", "firebrick"])

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, dpi=300)
        plt.subplots_adjust(bottom=0.25, top=0.85, wspace=0.04)
        axStars = axes[0]
        axGals = axes[1]
        axStars.set_aspect(1, anchor="C")
        axStars.tick_params(labelsize=8)
        axGals.set_aspect(1, anchor="C")
        axGals.tick_params(labelsize=8)

        # Need to separate stars and galaxies
        stars = (catPlot["sourceType"] == 1)
        gals = (catPlot["sourceType"] == 2)

        xCol = self.config.axisLabels["x"]
        yCol = self.config.axisLabels["y"]

        # For galaxies
        xsGals = catPlot.loc[gals, xCol]
        ysGals = catPlot.loc[gals, yCol]
        nGals = len(xsGals)

        # For stars
        xsStars = catPlot.loc[stars, xCol]
        ysStars = catPlot.loc[stars, yCol]
        nStars = len(xsStars)

        # Set useful axis limits
        if self.config.xLims is not None:
            axStars.set_xlim(self.config.xLims[0], self.config.xLims[1])
            axGals.set_xlim(self.config.xLims[0], self.config.xLims[1])
        else:
            if nStars > minPoints:
                starPercsX = np.nanpercentile(xsStars, [1, 99.5])
                pad = (starPercsX[1] - starPercsX[0])/3
                axStars.set_xlim(starPercsX[0] - pad, starPercsX[1] + pad)
                axGals.set_xlim(starPercsX[0] - pad, starPercsX[1] + pad)

        if self.config.yLims is not None:
            axStars.set_ylim(self.config.yLims[0], self.config.yLims[1])
            axGals.set_ylim(self.config.yLims[0], self.config.yLims[1])
        else:
            if nStars > minPoints:
                starPercsY = np.nanpercentile(ysStars, [1, 99.5])
                pad = (starPercsY[1] - starPercsY[0])/3
                axStars.set_ylim(starPercsY[0] - pad, starPercsY[1] + pad)
                axGals.set_ylim(starPercsY[0] - pad, starPercsY[1] + pad)

        # Get the axes positions to aid in positioning text and colorbar axes
        # below.
        axPosStars = axStars.get_position()
        axPosGals = axGals.get_position()
        xAxStarsMin = axPosStars.x0
        xAxStarsMax = axPosStars.x1
        xAxDelta = xAxStarsMax - xAxStarsMin
        yAxMin = axPosStars.y0
        yAxMax = axPosStars.y1
        xAxGalsMin = axPosGals.x0

        if nStars > minPoints:
            # Calculate the point density
            xyStars = np.vstack([xsStars, ysStars])
            starDens = scipyStats.gaussian_kde(xyStars)(xyStars)
            # Compute 2d histograms of the stars (used in the contour plot
            # function).
            countsStars, xEdgesStars, yEdgesStars = np.histogram2d(xsStars, ysStars, bins=self.config.nBins,
                                                                   normed=False)
            zsStars = countsStars.transpose()
            [vminStars, vmaxStars] = np.nanpercentile(zsStars, [1, 99])
            vminStars = max(5, vminStars)
            levelsStars = np.linspace(int(np.floor(vminStars)), int(np.ceil(vmaxStars)), num=nLevel)
            # Plot low density points
            axStars.scatter(xsStars[starDens <= maxDens], ysStars[starDens <= maxDens],
                            c="black", s=1, zorder=2.5)

        if nGals > minPoints:
            # Calculate the point density
            xyGals = np.vstack([xsGals, ysGals])
            galDens = scipyStats.gaussian_kde(xyGals)(xyGals)
            factor = nStars/nGals if nStars > minPoints else 1
            # Compute 2d histograms of the galaxies (used in the contour plot
            # function).
            countsGals, xEdgesGals, yEdgesGals = np.histogram2d(xsGals, ysGals, bins=self.config.nBins,
                                                                normed=False)
            zsGals = countsGals.transpose()
            [vminGals, vmaxGals] = np.nanpercentile(zsGals, [1, 99])
            vminGals = max(5, vminGals)
            levelsGals = np.linspace(int(np.floor(vminGals)), int(np.ceil(vmaxGals)),
                                     num=max(int(np.floor(nLevel*0.5*np.sqrt(nGals/nStars))), nLevel))
            # Plot low density points
            axGals.scatter(xsGals[galDens <= maxDens*factor], ysGals[galDens <= maxDens*factor],
                           c="black", s=1, zorder=2.5)

        if nStars > minPoints:
            # Add text details
            starBBox = dict(facecolor="paleturquoise", alpha=0.5, edgecolor="darkblue")
            fig.text(xAxStarsMin + 0.01, yAxMax + 0.02, "N Stars: {}".format(len(xsStars)),
                     bbox=starBBox, fontsize=7)
            starCpf = axStars.contourf(zsStars, cmap=newBlues, alpha=0.65, levels=levelsStars,
                                       extend="max", extent=[xEdgesStars.min(), xEdgesStars.max(),
                                                             yEdgesStars.min(), yEdgesStars.max()])
            starCp = axStars.contour(zsStars, cmap=newBlues, levels=levelsStars, extend="max",
                                     linewidths=2, extent=[xEdgesStars.min(), xEdgesStars.max(),
                                                           yEdgesStars.min(), yEdgesStars.max()])
            # Add colorbars
            starCbAx = fig.add_axes([axPosStars.x0, yAxMin - 0.16, xAxDelta, 0.04])
            starCbAx.tick_params(labelsize=7)
            starCb = plt.colorbar(starCpf, cax=starCbAx, extend="max", orientation="horizontal")
            starCb.add_lines(starCp)
            starText = starCbAx.text(0.5, 0.5, "Number Density: Stars", color="k", rotation="horizontal",
                                     transform=starCbAx.transAxes, ha="center", va="center", fontsize=7)
            starText.set_path_effects([pathEffects.Stroke(linewidth=2, foreground="w"),
                                       pathEffects.Normal()])

        if nGals > minPoints:
            # Add text details
            galBBox = dict(facecolor="lemonchiffon", alpha=0.5, edgecolor="firebrick")
            fig.text(xAxGalsMin + 0.01, yAxMax + 0.02, "N Galaxies: {}".format(len(xsGals)),
                     bbox=galBBox, fontsize=7)
            galCpf = axGals.contourf(zsGals, cmap=newReds, alpha=0.65, levels=levelsGals,
                                     extend="max", extent=[xEdgesGals.min(), xEdgesGals.max(),
                                                           yEdgesGals.min(), yEdgesGals.max()])
            galCp = axGals.contour(zsGals, cmap=newReds, levels=levelsGals, extend="max",
                                   linewidths=2, extent=[xEdgesGals.min(), xEdgesGals.max(),
                                                         yEdgesGals.min(), yEdgesGals.max()])
            # Add colorbars
            galCbAx = fig.add_axes([xAxGalsMin, yAxMin - 0.16, xAxDelta, 0.04])
            galCbAx.tick_params(labelsize=7)
            galCb = plt.colorbar(galCpf, cax=galCbAx, extend="max", orientation="horizontal")
            galCb.add_lines(galCp)
            galCbAx.yaxis.set_ticks_position("left")
            galText = galCbAx.text(0.5, 0.5, "Number Density: Galaxies", color="k", rotation="horizontal",
                                   transform=galCbAx.transAxes, ha="center", va="center", fontsize=7)
            galText.set_path_effects([pathEffects.Stroke(linewidth=2, foreground="w"),
                                      pathEffects.Normal()])

        axStars.set_xlabel(self.config.axisLabels["x"])
        axStars.set_ylabel(self.config.axisLabels["y"])
        axGals.set_xlabel(self.config.axisLabels["x"])

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
