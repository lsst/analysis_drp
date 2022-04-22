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

import lsst.pipe.base as pipeBase

from .dataSelectors import VisitPlotFlagSelector
from .rhoPlot import RhoPlotTask

__all__ = ["RhoPlotVisitTaskConfig", "RhoPlotVisitTask"]


class RhoPlotVisitTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "skymap"),
    defaultTemplates={"plotName": "deltaCoords"},
):

    catPlot = pipeBase.connectionTypes.Input(
        doc="The tract wide catalog to make plots from.",
        storageClass="DataFrame",
        name="sourceTable_visit",
        dimensions=("visit",),
        deferLoad=True,
    )

    rho0Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho0 statistics.",
        storageClass="Plot",
        name="rho0PlotVisit_{plotName}",
        dimensions=("visit",),
    )

    rho1Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho1 statistics.",
        storageClass="Plot",
        name="rho1PlotVisit_{plotName}",
        dimensions=("visit",),
    )

    rho2Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho2 statistics.",
        storageClass="Plot",
        name="rho2PlotVisit_{plotName}",
        dimensions=("visit",),
    )

    rho3Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho3 statistics.",
        storageClass="Plot",
        name="rho3PlotVisit_{plotName}",
        dimensions=("visit",),
    )

    rho4Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho4 statistics.",
        storageClass="Plot",
        name="rho4PlotVisit_{plotName}",
        dimensions=("visit",),
    )

    rho5Plot = pipeBase.connectionTypes.Output(
        doc="Plot with rho5 statistics.",
        storageClass="Plot",
        name="rho5PlotVisit_{plotName}",
        dimensions=("visit",),
    )


class RhoPlotVisitTaskConfig(RhoPlotTask.ConfigClass, pipelineConnections=RhoPlotVisitTaskConnections):
    """Configuration for RhoPlotVisitTask."""

    def _get_requirements(self):
        columnNames = set()
        bands = None

        for actionStruct in (self.selectorActions, self.sourceSelectorActions):
            for action in actionStruct:
                columnNames.update(action.columns)

        columnNames.update(self.rhoStatisticsAction.columns)

        return bands, columnNames

    def setDefaults(self):
        super().setDefaults()
        self.sourceSelectorActions.sourceSelector.band = ""
        self.selectorActions.sourceSelector = VisitPlotFlagSelector

        self.rhoStatisticsAction.treecorr.min_sep = 0.01
        self.rhoStatisticsAction.treecorr.max_sep = 100
        self.rhoStatisticsAction.treecorr.nbins = 15

        self.rhoStatisticsAction.colRa.column = "coord_ra"
        self.rhoStatisticsAction.colRa.inRadians = False
        self.rhoStatisticsAction.colDec.column = "coord_dec"
        self.rhoStatisticsAction.colDec.inRadians = False

        self.rhoStatisticsAction.colXx = "ixx"
        self.rhoStatisticsAction.colYy = "iyy"
        self.rhoStatisticsAction.colXy = "ixy"
        self.rhoStatisticsAction.colPsfXx = "ixxPSF"
        self.rhoStatisticsAction.colPsfYy = "iyyPSF"
        self.rhoStatisticsAction.colPsfXy = "ixyPSF"


class RhoPlotVisitTask(RhoPlotTask):
    """A task to make rho plots at visit level"""

    ConfigClass = RhoPlotVisitTaskConfig
    _DefaultName = "rhoPlotVisitTask"
