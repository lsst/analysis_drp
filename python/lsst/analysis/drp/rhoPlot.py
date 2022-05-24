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

import lsst.pipe.base as pipeBase
from lsst.pex.config import ChoiceField, DictField, Field
from lsst.pipe.tasks.configurableActions import ConfigurableActionField, ConfigurableActionStructField

from . import dataSelectors
from .calcFunctors import CalcRhoStatistics
from .plotUtils import addPlotInfo, parsePlotInfo

__all__ = ["RhoPlotTaskConfig", "RhoPlotTask"]


class RhoPlotTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap"),
    defaultTemplates={"inputCoaddName": "deep", "plotName": "deltaCoords"},
):

    catPlot = pipeBase.connectionTypes.Input(
        doc="The tract wide catalog to make plots from.",
        storageClass="DataFrame",
        name="objectTable_tract",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )

    rho0Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho0 statistics.",
        storageClass="Plot",
        name="rho0Plot_{plotName}",
        dimensions=("tract", "skymap"),
    )

    rho1Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho1 statistics.",
        storageClass="Plot",
        name="rho1Plot_{plotName}",
        dimensions=("tract", "skymap"),
    )

    rho2Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho2 statistics.",
        storageClass="Plot",
        name="rho2Plot_{plotName}",
        dimensions=("tract", "skymap"),
    )

    rho3Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho3 statistics.",
        storageClass="Plot",
        name="rho3Plot_{plotName}",
        dimensions=("tract", "skymap"),
    )

    rho4Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho4 statistics.",
        storageClass="Plot",
        name="rho4Plot_{plotName}",
        dimensions=("tract", "skymap"),
    )

    rho5Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho5 statistics.",
        storageClass="Plot",
        name="rho5Plot_{plotName}",
        dimensions=("tract", "skymap"),
    )


class RhoPlotTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=RhoPlotTaskConnections):
    """Configuration for RhoPlotTask."""

    rhoStatisticsAction = ConfigurableActionField(
        doc="The action that computes the Rho statistics",
        default=CalcRhoStatistics,
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={
            "flagSelector": dataSelectors.CoaddPlotFlagSelector,
            "sourceSelector": dataSelectors.StarIdentifier,
        },
    )

    sourceSelectorActions = ConfigurableActionStructField(
        doc="What types of sources to use.",
        default={"sourceSelector": dataSelectors.StarIdentifier},
    )

    xScale = ChoiceField(
        doc="The scale to use for the x-axis.",
        default="log",
        dtype=str,
        allowed={scale: scale for scale in ("linear", "log")},
    )

    yScale = ChoiceField(
        doc="The scale to use for the y-axis.",
        default="symlog",
        dtype=str,
        allowed={scale: scale for scale in ("linear", "log", "symlog")},
    )

    linthresh = Field(
        doc=(
            "The value around zero where the scale becomes linear in y-axis "
            "when symlog is set as the scale. Sets the `linthresh` parameter "
            "of `~matplotlib.axes.set_yscale`."
        ),
        default=1e-6,
        dtype=float,
    )

    yAxisLabels = DictField(
        doc=(
            "The labels for the y-axis. The labels are stored in a dictionary "
            "with the keys in the range 0-6 and the values being the labels."
        ),
        keytype=int,
        itemtype=str,
    )

    yAxisLimMin = DictField(
        doc="ymin values for individual RhoStatistics plots.",
        keytype=int,
        itemtype=float,
        default={k: None for k in range(6)},
    )

    yAxisLimMax = DictField(
        doc="ymax values for individual RhoStatistics plots.",
        keytype=int,
        itemtype=float,
        default={k: None for k in range(6)},
    )

    def _get_requirements(self):
        columnNames = set(["patch"])
        bands = set()

        for actionStruct in (self.selectorActions, self.sourceSelectorActions):
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)
                    band = col.split("_")[0]
                    if band not in ("coord", "extend", "detect", "xy", "merge"):
                        bands.add(band)

        columnNames.update(self.rhoStatisticsAction.columns)

        return bands, columnNames

    def setDefaults(self):
        super().setDefaults()

        self.rhoStatisticsAction.treecorr.min_sep = 0.01
        self.rhoStatisticsAction.treecorr.max_sep = 80
        self.rhoStatisticsAction.treecorr.nbins = 21

        self.rhoStatisticsAction.colRa.column = "coord_ra"
        self.rhoStatisticsAction.colRa.inRadians = False
        self.rhoStatisticsAction.colDec.column = "coord_dec"
        self.rhoStatisticsAction.colDec.inRadians = False

        self.rhoStatisticsAction.colXx = "i_ixx"
        self.rhoStatisticsAction.colYy = "i_iyy"
        self.rhoStatisticsAction.colXy = "i_ixy"
        self.rhoStatisticsAction.colPsfXx = "i_ixxPSF"
        self.rhoStatisticsAction.colPsfYy = "i_iyyPSF"
        self.rhoStatisticsAction.colPsfXy = "i_ixyPSF"

        _yLabels = {
            0: r"$\rho_{0}(\theta) = \langle \frac{\delta T}{T}, \frac{\delta T}{T}\rangle$",
            1: r"$\rho_{1}(\theta) = \langle \delta e, \delta e \rangle$",
            2: r"$\rho_{2}(\theta) = \langle e, \delta e \rangle$",
            3: r"$\rho_{3}(\theta) = \langle e\frac{\delta T}{T} , e\frac{\delta T}{T} \rangle$",
            4: r"$\rho_{4}(\theta) = \langle \delta e, e\frac{\delta T}{T} \rangle$",
            5: r"$\rho_{5}(\theta) = \langle e, e\frac{\delta T}{T} \rangle$",
        }

        self.yAxisLabels = _yLabels


class RhoPlotTask(pipeBase.PipelineTask):

    ConfigClass = RhoPlotTaskConfig
    _DefaultName = "rhoPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        bands, columnNames = self.config._get_requirements()

        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs["catPlot"] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        inputs["bands"] = bands if bands else dataId["band"]
        outputs = self.run(**inputs)
        # TODO: DM-34939 Remove the conditional butler.put
        if outputs:
            butlerQC.put(outputs, outputRefs)
            plt.close()

    def run(self, catPlot, dataId, runName, tableName, bands):
        """Run the rho-statistics computation and plot them.

        Parameters
        ----------
        catPlot : `~pandas.core.frame.DataFrame`
            The catalog to compute the rho-statistics from.
        dataId: `~lsst.daf.butler.core.dimensions._coordinate._ExpandedTupleDataCoordinate`  # noqa
            The dimensions that the plot is being made from.
        runName : `str`
            The name of the collection that the plot is written out to.
        tableName : `str`
            The type of table used to make the plot.
        bands : `set`
            The set of bands from which data is taken.

        Returns
        -------
        `pipeBase.Struct` containing 6 rho-statistics plots.

        Notes
        -----
        The catalogue is first narrowed down using the selectors specified in
        `self.config.selectorActions`.
        If the column names are 'Functor' then the functors specified in
        `self.config.axisFunctors` are used to calculate the required values.
        After this the following functions are run:

        `parsePlotInfo` which uses the dataId, runName and tableName to add
        useful information to the plot.

        `rhoPlot` makes six plots for six different rho-statistics.
        """

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot).astype(bool)
        catPlot = catPlot[mask]

        columns = {}
        try:
            columns["patch"] = catPlot["patch"]
        except KeyError:
            pass

        for action in self.config.sourceSelectorActions:
            for col in action.columns:
                columns.update({col: catPlot[col]})

        # Check the columns have finite values
        mask = np.ones(len(catPlot), dtype=bool)
        for col in catPlot.columns:
            mask &= np.isfinite(catPlot[col])
        catPlot = catPlot[mask]

        if len(catPlot) < 2:
            self.log.error("Not enough sources to make Rho statistics plots")
            # TODO: DM-34939 Raise ValueError after fixing the breakage
            # in ci_hsc
            return None

        # This should be unnecessary as we will always select stars
        sourceTypes = np.zeros(len(catPlot))
        for selector in self.config.sourceSelectorActions:
            # The source selectors return 1 for a star and 2 for a galaxy
            # rather than a mask this allows the information about which
            # type of sources are being plotted to be propagated
            sourceTypes += selector(catPlot)
        if list(self.config.sourceSelectorActions) == []:
            sourceTypes = [10] * len(catPlot)
        catPlot.loc[:, "sourceType"] = sourceTypes

        # Get the S/N cut used
        try:
            SN = self.config.selectorActions.SnSelector.threshold
        except AttributeError:
            SN = "N/A"

        rhoStat = self.config.rhoStatisticsAction(catPlot)

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, bands, None, SN)

        # Make the plot(s)
        figDict = self.rhoPlot(rhoStat, plotInfo)

        return pipeBase.Struct(rho0Plot=figDict[0], rho1Plot=figDict[1], rho2Plot=figDict[2],
                               rho3Plot=figDict[3], rho4Plot=figDict[4], rho5Plot=figDict[5])

    def rhoPlot(self, rhoStats, plotInfo):
        """Makes a plot for each rho-statistic in ``rhoStats``.

        Parameters
        ----------
        rhoStats: `dict` [`int`, `treecorr.BinnerCorr2`]
            A dictionary with the rho-statistic index as the key and the
            corresponding correlation function as the value.
        plotInfo: `dict`
            A dictionary of information about the data being plotted with keys:
                ``"run"``
                    The output run for the plots (`str`).
                ``"skymap"``
                    The type of skymap used for the data (`str`).
                ``"filter"``
                    The filter used for this data (`str`).
                ``"tract"``
                    The tract that the data comes from (`str`).

        Returns
        -------
        figDict: `dict` [`int`, `~matplotlib.figure.Figure`]
            A list of figures containing the rho-statistic plots.
        """
        figDict = {}
        for rhoIndex in rhoStats:
            rho = rhoStats[rhoIndex]
            # KK/GGCorrelation objects have different entries for variance in x
            try:
                xi = rho.xip
                yErr = np.sqrt(rho.varxip)
            except AttributeError:
                xi = rho.xi
                yErr = np.sqrt(rho.varxi)

            self.log.verbose("The values of rho %s are %s", rhoIndex, xi)

            isPositive = xi > 0

            # If the y-axis is log scale, plot the absolute value of the
            # rho statistics and draw the negative points as unfilled.
            if self.config.yScale == "log":
                fillstyle, label, sgn = "none", "Negative values", -1
            else:
                fillstyle, label, sgn = None, None, 1

            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            ax.errorbar(rho.meanr[isPositive], xi[isPositive], yerr=yErr[isPositive],
                        color="C0", fmt="o")
            ax.errorbar(rho.meanr[~isPositive], sgn*xi[~isPositive], yerr=yErr[~isPositive],
                        color="C0", fmt="o", fillstyle=fillstyle, label=label)

            ax.set_xscale(self.config.xScale)
            ax.set_ylim(self.config.yAxisLimMin[rhoIndex], self.config.yAxisLimMax[rhoIndex])

            if self.config.yScale == "symlog":
                linthresh = self.config.linthresh
                ax.set_yscale("symlog", linthresh=linthresh)
                if min(ax.get_ylim()) < linthresh:
                    ax.axhline(0, color="k", linestyle="--")
                    ax.axhspan(-linthresh, linthresh, color="gray", alpha=0.2, label="Linear scale region")
                    ax.legend(loc="upper right")
            else:
                ax.set_yscale(self.config.yScale)
                if self.config.yScale == "log" and sum(~isPositive) > 0:
                    ax.legend(loc="upper right")

            ax.set_ylabel(self.config.yAxisLabels[rhoIndex])

            units = self.config.rhoStatisticsAction.treecorr.sep_units
            ax.set_xlabel(rf"$\theta$ ({units})")

            plotInfo["plotName"] = f"rho{rhoIndex}_deltaCoords"
            fig = addPlotInfo(fig, plotInfo)
            fig.tight_layout()

            figDict[rhoIndex] = fig

        return figDict
