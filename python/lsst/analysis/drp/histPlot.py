import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import SingleColumnAction
from lsst.skymap import BaseSkyMap
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from . import dataSelectors
from . import calcFunctors
from .plotUtils import generateSummaryStats, parsePlotInfo, addPlotInfo


class HistPlotTaskConnections(pipeBase.PipelineTaskConnections,
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

    histPlot = pipeBase.connectionTypes.Output(doc="A two-panel plot showing histograms for specified data.",
                                               storageClass="Plot",
                                               name="histPlot_{plotName}",
                                               dimensions=("tract", "skymap"))


class HistPlotTaskConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=HistPlotTaskConnections):

    leftPanelActions = ConfigurableActionStructField(
        doc="The actions used to calculate the values for each histogram in the left panel.",
        default={"hist1": SingleColumnAction,
                 "hist2": SingleColumnAction},
    )

    rightPanelActions = ConfigurableActionStructField(
        doc="The actions used to calculate the values for each histogram in the right panel.",
        default={"hist1": calcFunctors.SNCalculator,
                 "hist2": calcFunctors.SNCalculator},
    )

    leftPanelLabels = pexConfig.DictField(
        doc="The labels associated with each histogram in the left panel. These dict keys should match "
        "those in leftPanelActions. If no match is found, a placeholder label will be assigned instead.",
        keytype=str,
        itemtype=str,
        default={"hist1": "PSFlux", "hist2": "9ApFlux"},
    )

    rightPanelLabels = pexConfig.DictField(
        doc="The labels associated with each histogram in the right panel. These dict keys should match "
        "those in rightPanelActions. If no match is found, a placeholder label will be assigned instead.",
        keytype=str,
        itemtype=str,
        default={"hist1": "PSFlux SN", "hist2": "9ApFlux SN"},
    )

    axisLabels = pexConfig.DictField(
        doc="Axis labels for both x-axes, and one for the single unified y-axis.",
        keytype=str,
        itemtype=str,
        default={"xLeft": "flux (nJy)", "xRight": "S/N", "yLeft": "frequency"},
    )

    summaryStatsLabel = pexConfig.Field(
        doc="Name of the column used to generate the on-sky summary plot. Should be one of the keys in "
        "leftPanelLabels/rightPanelLabels. If no match is found, set to the first key in leftPanelLabels.",
        dtype=str,
        default="9ApFlux",
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": dataSelectors.FlagSelector},
    )

    pLower = pexConfig.Field(
        doc="Percentile used to determine the lower range of the histogram bins. If more than one histogram "
        "is plotted per-panel, the percentile limit is the minimum value across all input data.",
        dtype=float,
        default=2.0,
    )

    pUpper = pexConfig.Field(
        doc="Percentile used to determine the upper range of the histogram bins. If more than one histogram "
        "is plotted per-panel, the percentile limit is the maximum value across all input data.",
        dtype=float,
        default=98.0,
    )

    nBins = pexConfig.Field(
        doc="Number of bins used to divide the x axes.",
        default=40,
        dtype=int,
    )

    def setDefaults(self):
        super().setDefaults()
        self.leftPanelActions.hist1.column = "i_psfFlux"
        self.leftPanelActions.hist2.column = "i_ap09Flux"
        self.rightPanelActions.hist1.colA.column = "i_psfFlux"
        self.rightPanelActions.hist1.colB.column = "i_psfFluxErr"
        self.rightPanelActions.hist2.colA.column = "i_ap09Flux"
        self.rightPanelActions.hist2.colB.column = "i_ap09FluxErr"
        # assign dummy labels if any are missing
        used_labels = []
        for key in self.leftPanelActions.fieldNames:
            if key not in self.leftPanelLabels.keys():
                self.leftPanelLabels.update({key: f"left_{key}"})
            used_labels.append(self.leftPanelLabels[key])
        for key in self.rightPanelActions.fieldNames:
            if key not in self.rightPanelLabels.keys():
                self.rightPanelLabels.update({key: f"right_{key}"})
            used_labels.append(self.rightPanelLabels[key])
        if self.summaryStatsLabel not in used_labels:
            self.summaryStatsLabel = used_labels[0]


class HistPlotTask(pipeBase.PipelineTask):

    ConfigClass = HistPlotTaskConfig
    _DefaultName = "histPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        columnNames = set(["patch"])
        bands = []
        for actionStruct in [self.config.leftPanelActions,
                             self.config.rightPanelActions,
                             self.config.selectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)
                    band = col.split("_")[0]
                    if band not in ["coord", "extend", "detect", "xy"]:
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
        inputs["plotName"] = localConnections.histPlot.name
        inputs["bands"] = bands
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, skymap, tableName, bands, plotName):
        """Prep the catalogue and then make a two-panel plot showing various
        histograms in each panel.

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

        columns = {"patch": catPlot["patch"]}

        for key in self.config.leftPanelActions.fieldNames:
            label = self.config.leftPanelLabels[key]
            action = getattr(self.config.leftPanelActions, key)
            columns[label] = action(catPlot)

        for key in self.config.rightPanelActions.fieldNames:
            label = self.config.rightPanelLabels[key]
            action = getattr(self.config.rightPanelActions, key)
            columns[label] = action(catPlot)

        plotDf = pd.DataFrame(columns)

        # Get the S/N cut used
        try:
            SN = self.config.selectorActions.SnSelector.threshold
        except AttributeError:
            SN = "N/A"

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, plotName, SN)
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStats(plotDf, self.config.summaryStatsLabel, skymap, plotInfo)
        # Make the plot
        fig = self.histPlot(plotDf, plotInfo, sumStats)

        return pipeBase.Struct(histPlot=fig)

    def histPlot(self, catPlot, plotInfo, sumStats):

        """Makes a two-panel plot with histograms of data displayed in each
        panel.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        plotInfo : `dict`
            A dictionary of information about the data being plotted with keys:
                `"run"`
                    The output run for the plots (`str`).
                `"skymap"`
                    The type of skymap used for the data (`str`).
                `"filter"`
                    The filter used for this data (`str`).
                `"tract"`
                    The tract that the data comes from (`str`).
        sumStats : `dict`
            A dictionary where the patchIds are the keys which store the R.A.
            and dec of the corners of the patch, along with a summary
            statistic for each patch.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        A summary panel showing the median of the summaryStatsLabel in each
        patch is shown in the upper right corner of the resultant plot. The
        code uses the selectorActions to decide which points to plot and the
        statisticSelector actions to determine which points to use for the
        printed statistics.
        """
        self.log.info(f"Plotting {self.config.connections.plotName} on a two-panel histogram plot: "
                      f"panel 1 = {self.config.leftPanelLabels}; "
                      f"panel 2 = {self.config.rightPanelLabels} ")

        fig = plt.figure(dpi=200)
        gs = gridspec.GridSpec(100, 100)

        # left panel limits
        pLowers, pUppers = [], []
        for key in self.config.leftPanelActions.fieldNames:
            column = self.config.leftPanelLabels[key]
            [pLower, pUpper] = np.nanpercentile(catPlot[column].values,
                                                [self.config.pLower, self.config.pUpper])
            pLowers.append(pLower)
            pUppers.append(pUpper)
        leftPanelLower = np.min(pLowers)
        leftPanelUpper = np.max(pUppers)

        # left panel plotting
        leftPanelAx = fig.add_subplot(gs[:, 0:34])
        for count, key in enumerate(self.config.leftPanelActions.fieldNames):
            column = self.config.leftPanelLabels[key]
            mask = np.isfinite(catPlot[column].values)
            label = f"{column} ({np.sum(mask)})"
            col = plt.cm.tab10(count)
            leftPanelAx.hist(catPlot[column][mask], bins=self.config.nBins, alpha=0.7, color=col,
                             range=(leftPanelLower, leftPanelUpper), histtype="bar", lw=2, label=label)
            datMed = np.nanmedian(catPlot[column][mask])
            datMad = sigmaMad(catPlot[column][mask])
            leftPanelAx.text(x=0.03, y=0.985-(count*0.08), transform=leftPanelAx.transAxes,
                             ha="left", va="top", fontsize=7, c=col,
                             s=f"med = {datMed:0.2f}\n$σ_{{MAD}}$ = {datMad:0.2f}")
            leftPanelAx.axvline(datMed, ls=':', lw=2, c=col)
        leftPanelAx.set_xlim(leftPanelLower, leftPanelUpper)
        leftPanelAx.set_xlabel(self.config.axisLabels["xLeft"])
        leftPanelAx.set_ylabel(self.config.axisLabels["yLeft"])
        leftPanelAx.legend(loc="upper left", bbox_to_anchor=(0.76, 0.55),
                           bbox_transform=fig.transFigure, fontsize=7, handleheight=1.5)
        leftPanelAx.axvline(0, ls='--', lw=2, c='k')

        # right panel limits
        pLowers, pUppers = [], []
        for key in self.config.rightPanelActions.fieldNames:
            column = self.config.rightPanelLabels[key]
            [pLower, pUpper] = np.nanpercentile(catPlot[column].values,
                                                [self.config.pLower, self.config.pUpper])
            pLowers.append(pLower)
            pUppers.append(pUpper)
        rightPanelLower = np.min(pLowers)
        rightPanelUpper = np.max(pUppers)

        # right panel plotting
        rightPanelAx = fig.add_subplot(gs[:, 42:76])
        plt.rcParams["hatch.color"] = "white"
        for count, key in enumerate(self.config.rightPanelActions.fieldNames):
            column = self.config.rightPanelLabels[key]
            mask = np.isfinite(catPlot[column].values)
            label = f"{column} ({np.sum(mask)})"
            col = plt.cm.tab10(count)
            rightPanelAx.hist(catPlot[column][mask], bins=self.config.nBins, alpha=0.7, color=col,
                              range=(rightPanelLower, rightPanelUpper), histtype="bar", lw=2, label=label,
                              hatch="//")
            datMed = np.nanmedian(catPlot[column][mask])
            datMad = sigmaMad(catPlot[column][mask])
            rightPanelAx.text(x=0.03, y=0.985-(count*0.08), transform=rightPanelAx.transAxes,
                              ha="left", va="top", fontsize=7, c=col,
                              s=f"med = {datMed:0.2f}\n$σ_{{MAD}}$ = {datMad:0.2f}")
            rightPanelAx.axvline(datMed, ls=':', lw=2, c=col)
        rightPanelAx.set_xlim(rightPanelLower, rightPanelUpper)
        rightPanelAx.set_xlabel(self.config.axisLabels["xRight"])
        yStep = 0.038*len(self.config.leftPanelActions.fieldNames) + 0.01
        rightPanelAx.legend(loc="upper left", bbox_to_anchor=(0.76, 0.55-yStep),
                            bbox_transform=fig.transFigure, fontsize=7, handleheight=1.5)
        rightPanelAx.axvline(0, ls='--', lw=2, c='k')

        # Corner plot of patches showing summary stat in each
        axCorner = plt.gcf().add_subplot(gs[0:26, -20:-3])
        axCorner.yaxis.tick_right()
        axCorner.yaxis.set_label_position("right")
        axCorner.xaxis.tick_top()
        axCorner.xaxis.set_label_position("top")

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
                axCorner.annotate(dataId, (cenX, cenY), color="k", fontsize=5, ha="center", va="center")

        cmapUse = plt.cm.coolwarm
        # Set the bad color to transparent and make a masked array
        cmapUse.set_bad(color="none")
        colors = np.ma.array(colors, mask=np.isnan(colors))
        collection = PatchCollection(patches, cmap=cmapUse)
        collection.set_array(colors)
        axCorner.add_collection(collection)

        axCorner.set_xlabel("R.A. (deg)", fontsize=8)
        axCorner.set_ylabel("Dec. (deg)", fontsize=8)
        axCorner.tick_params(axis="both", labelsize=8)

        # Add a colorbar
        divider = make_axes_locatable(axCorner)
        cax = divider.append_axes("bottom", size="14%")
        cbar = plt.colorbar(collection, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        cax.yaxis.set_ticks_position("left")
        for label in cax.yaxis.get_ticklabels():
            label.set_bbox(dict(facecolor="w", ec="none", alpha=0.5))
            label.set_fontsize(8)
        cax.text(0.5, 0.42, "Median Value", color="k", rotation="horizontal", transform=cax.transAxes,
                 horizontalalignment="center", verticalalignment="center", fontsize=8)
        axCorner.text(0.5, -0.5, self.config.summaryStatsLabel, color="k", rotation="horizontal",
                      transform=axCorner.transAxes, horizontalalignment="center", verticalalignment="center",
                      fontsize=9)

        plt.draw()

        # Add useful information to the plot
        plt.subplots_adjust(right=0.95)
        fig = plt.gcf()

        fig = addPlotInfo(fig, plotInfo)

        return fig
