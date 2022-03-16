import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.skymap import BaseSkyMap
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

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


class HistPlotConfig(pexConfig.Config):

    label = pexConfig.Field(
        doc="The x-axis label for the panel.",
        dtype=str,
        default="x axis",
    )

    actions = ConfigurableActionStructField(
        doc="A dict of configurable actions, with each key-value pair corresponding to each histogram. "
        "Dict keys will be used to label each histogram in the panel.",
        default={},
    )

    selectors = ConfigurableActionStructField(
        doc="Panel data selectors used to further narrow down the data. This is in addition to any global "
        "selectors specified in `selectorActions`. Selectors will be matched to their corresponding action "
        "using their dict key.",
        default={},
    )

    yscale = pexConfig.Field(
        doc="The scaling on the panel y-axis.",
        dtype=str,
        default="linear",
    )

    pLower = pexConfig.Field(
        doc="Percentile used to determine the lower range of the histogram bins. If more than one histogram "
        "is plotted, the percentile limit is the minimum value across all input data.",
        dtype=float,
        default=2.0,
    )

    pUpper = pexConfig.Field(
        doc="Percentile used to determine the upper range of the histogram bins. If more than one histogram "
        "is plotted, the percentile limit is the maximum value across all input data.",
        dtype=float,
        default=98.0,
    )

    nBins = pexConfig.Field(
        doc="Number of bins used to divide the x axis into.",
        default=50,
        dtype=int,
    )


class HistPlotTaskConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=HistPlotTaskConnections):

    panels = pexConfig.ConfigDictField(
        doc="A configurable dict describing the panels to be plotted, and the histograms for each panel.",
        keytype=str,
        itemtype=HistPlotConfig,
        default={},
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data. These selectors are applied globally to all "
        "panels in the plot. Per-histogram selectors may also be applied within the 'panels' argument.",
        default={},
    )

    summaryStatsColumn = pexConfig.Field(
        doc="Name of the column used to generate the on-sky summary plot.",
        dtype=str,
        default="i_ap09Flux",
    )


class HistPlotTask(pipeBase.PipelineTask):

    ConfigClass = HistPlotTaskConfig
    _DefaultName = "histPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        # Identify columns to extract from the butler
        selectorColumns = [col for selector in self.config.selectorActions for col in selector.columns]
        columnNames = set(["patch", self.config.summaryStatsColumn] + selectorColumns)
        bands = set([])
        for panel in self.config.panels:
            for action in self.config.panels[panel].actions:
                acols = [x for x in action.columns]
                columnNames.update(acols)
                for acol in acols:
                    band = acol.split("_")[0]
                    if band not in ["coord", "extend", "detect", "xy", "merge"]:
                        bands.update(band)
            for selector in self.config.panels[panel].selectors:
                scols = [x for x in selector.columns]
                columnNames.update(scols)

        # Get a reduced catalogue, generate inputs for the run method
        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs["catPlot"] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        inputs["plotName"] = localConnections.histPlot.name
        inputs["bands"] = bands

        # Run the task, put the results into repo using the butler
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
        bands : `set`
            The bands used to generate this figure.
        plotName : `str`
            The output plot name.

        Returns
        -------
        `pipeBase.Struct` containing:
            scatterPlot : `matplotlib.figure.Figure`
                The resulting figure.

        Notes
        -----
        The catalogue is first narrowed down using the selectors specified in
        `self.config.selectorActions`. Further data selections are made on a
        per-histogram basis with the `self.config.panels[].selectors`.
        If the column names are 'Functor' then the functors specified in
        `self.config.axisFunctors` are used to calculate the required values.
        After this the following functions are run:

        `parsePlotInfo` which uses the dataId, runName and tableName to add
        useful information to the plot.

        `generateSummaryStats` which parses the skymap to give the corners of
        the patches for later plotting and calculates some basic statistics
        in each patch for the column in self.zColName.

        `histPlot` which makes an N-panel figure containing a series of
        histograms.
        """

        # Apply the global selectors to narrow down the objects to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask]

        # Make a plot data dataframe
        plotDf = pd.DataFrame(catPlot)

        # Process all actions to make results columns, ready for plotting
        for i, panel in enumerate(self.config.panels):
            hists = [x for x in self.config.panels[panel].toDict()["actions"].keys()]
            actions = self.config.panels[panel].actions
            for hist, action in zip(hists, actions):
                plotDf[f"p{i}_{hist}"] = np.array(action(catPlot))

        # Gather useful information about the plot
        try:
            SN = self.config.selectorActions.SnSelector.threshold
        except AttributeError:
            SN = "N/A"
        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, plotName, SN)
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStats(plotDf, self.config.summaryStatsColumn, skymap, plotInfo)

        # Make the plot
        fig = self.histPlot(plotDf, plotInfo, sumStats)

        return pipeBase.Struct(histPlot=fig)

    def histPlot(self, catPlot, plotInfo, sumStats):

        """Make an N-panel plot with a user-configurable number of histograms
        displayed in each panel.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        plotInfo : `dict`
            A dictionary of information about the data being plotted with keys:
                `"run"`
                    Output run for the plots (`str`).
                `"tractTableType"`
                    Table from which results are taken (`str`).
                `"plotName"`
                    Output plot name (`str`)
                `"SN"`
                    The global signal-to-noise data threshold (`float`)
                `"skymap"`
                    The type of skymap used for the data (`str`).
                `"tract"`
                    The tract that the data comes from (`int`).
                `"bands"`
                    The bands used for this data (`str` or `list`).
                `"visit"`
                    The visit that the data comes from (`int`)

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
        A summary panel showing the median of the summaryStatsColumn in each
        patch is shown in the upper right corner of the resultant plot.
        """
        panels = dict(self.config.panels.items())
        self.log.info(f"Generating a {len(panels)}-panel histogram plot.")

        fig = plt.figure(dpi=300)
        gs = gridspec.GridSpec(240, 240)

        # Determine gridspec figure divisions
        summary_block = 40  # how much space should be reserved for the summary stats block?
        panel_pad = 8  # how much padding should be added around each panel?
        if len(panels) <= 1:
            ncols = 1
        else:
            ncols = 2
        nrows = int(np.ceil(len(panels) / ncols))
        col_bounds = np.linspace(0 + summary_block + panel_pad, 240, ncols + 1, dtype=int)
        row_bounds = np.linspace(0, 240, nrows + 1, dtype=int)
        col_starts, col_stops, row_starts, row_stops = [], [], [], []
        for i in range(len(panels)):
            col_starts.append(col_bounds[i % ncols] + panel_pad)
            if (i == (len(panels) - 1)) and (len(panels) % 2 == 1):
                col_stops.append(np.max(col_bounds) - panel_pad)
            else:
                col_stops.append(col_bounds[(i % ncols) + 1] - panel_pad)
            row_starts.append(row_bounds[i // ncols] + panel_pad)
            row_stops.append(row_bounds[(i // ncols) + 1] - (2 * panel_pad))

        # panel plotting
        for i, (panel, col_start, col_stop, row_start, row_stop) in enumerate(zip(panels,
                                                                                  col_starts, col_stops,
                                                                                  row_starts, row_stops)):
            hists = [hist for hist in panels[panel].toDict()["actions"].keys()]

            # get per-histogram column data
            hist_columns = dict()
            vLowers, vUppers, meds, mads, nums = [], [], [], [], []
            for hist in hists:
                column = f"p{i}_{hist}"  # unique histogram column name
                # apply per-histogram selector, if any requested
                hist_mask = np.ones(len(catPlot), dtype=bool)
                if hist in panels[panel].toDict()["selectors"].keys():
                    selector = getattr(panels[panel].selectors, hist)
                    hist_mask &= selector(catPlot) > 0
                # trim catPlot dataframe to selector rows only
                hist_data = catPlot[hist_mask][column]
                hist_columns.update({column: hist_data})
                # find histogram data lower/upper percentile limits
                pvals = np.nanpercentile(hist_data,
                                         [self.config.panels[panel].pLower, self.config.panels[panel].pUpper])
                vLowers.append(pvals[0])
                vUppers.append(pvals[1])
                # generate additional per-histogram statistics
                isfinite = np.isfinite(hist_data)
                meds.append(np.median(hist_data[isfinite]))
                mads.append(sigmaMad(hist_data[isfinite]))
                nums.append(np.sum(isfinite))
            vLower = np.min(vLowers)
            vUpper = np.max(vUppers)

            # generate plot
            ax = fig.add_subplot(gs[row_start:row_stop, col_start:col_stop])
            for count, (hist, hist_data, med) in enumerate(zip(hists, hist_columns.values(), meds)):
                col = plt.cm.tab10(count)
                ax.hist(hist_data[np.isfinite(hist_data)], bins=self.config.panels[panel].nBins, alpha=0.7,
                        color=col, range=(vLower, vUpper), histtype="step", lw=2)
                ax.axvline(med, ls="--", lw=1, c=col)
            ax.set_yscale(self.config.panels[panel].yscale)
            ax.set_xlim(vLower, vUpper)
            ax.set_xlabel(self.config.panels[panel].label, labelpad=1)
            ax.tick_params(labelsize=7)
            # add a buffer to the top of the plot to allow space for labels
            ylims = list(ax.get_ylim())
            if ax.get_yscale() == "log":
                ylims[1] = 10**(np.log10(ylims[1]) * 1.1)
            else:
                ylims[1] *= 1.1
            ax.set_ylim(ylims[0], ylims[1])

            # add histogram labels and data statistics
            for count, (hist, med, mad, num) in enumerate(zip(hists, meds, mads, nums)):
                stats = f"{med:0.1f}, {mad:0.1f}, {num}"
                ax.text(0.01, 0.99, "\n"*count+hist, c=plt.cm.tab10(count), fontsize=7, ha="left",
                        va="top", transform=ax.transAxes)
                ax.text(0.99, 0.99, "\n"*count+stats, c=plt.cm.tab10(count), fontsize=7, ha="right",
                        va="top", transform=ax.transAxes)

        # summary stats plot
        axCorner = fig.add_subplot(gs[-summary_block-59:, :summary_block])
        axCorner.set_aspect("equal")
        axCorner.invert_xaxis()
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
        axCorner.set_xlabel("R.A. (deg)", fontsize=7, labelpad=1)
        axCorner.set_ylabel("Dec. (deg)", fontsize=7, labelpad=1)
        axCorner.tick_params(labelsize=5, length=2, pad=1)
        # add a colorbar
        divider = make_axes_locatable(axCorner)
        cax = divider.append_axes("top", size="14%", pad=0.05)
        cbar = plt.colorbar(collection, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=5, labeltop=True, labelbottom=False, top=True, bottom=False, length=2,
                            pad=0.5)
        cax.text(0.5, 0.4, "Median Value", color="k", rotation="horizontal", transform=cax.transAxes,
                 horizontalalignment="center", verticalalignment="center", fontsize=7)
        axCorner.text(0.5, 1.35, self.config.summaryStatsColumn, color="k", rotation="horizontal",
                      transform=axCorner.transAxes, horizontalalignment="center", verticalalignment="center",
                      fontsize=7)

        # Wrap up: add global y-axis label, hist stats key, and adjust subplots
        plt.text((summary_block + panel_pad / 2) / 240, 0.41, "Frequency", rotation=90,
                 transform=fig.transFigure)
        plt.text(0.955, 0.889, "Key: med, ${{\\sigma}}_{{MAD}}$, $n_{{points}}$",
                 transform=fig.transFigure, ha="right", va="bottom", fontsize=7)
        plt.draw()
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.02, top=0.91)
        fig = addPlotInfo(fig, plotInfo)

        return fig
