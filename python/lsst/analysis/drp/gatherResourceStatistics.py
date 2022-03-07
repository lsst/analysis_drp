# This file is part of analysis_drp.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import defaultdict

__all__ = ()

from lsst.pex.config import ListField
from lsst.pipe.base import PipelineTaskConnections, PipelineTask, PipelineTaskConfig
from lsst.pipe.base import connectionTypes as ct


class GatherResourceStatisticsConnections(PipelineTaskConnections, dimensions=()):
    output_table = ct.Output(
        "resource_statistics",
        storageClass="DataFrame",
        dimensions=(),
        doc=(
            "Table that aggregates memory and CPU usage statistics from one "
            "or more tasks. "
            "This will have one row for each data ID, with columns for each "
            "task or method's memory usage and runtime."
        )
    )

    def __init__(self, *, config):
        super().__init__(config=config)
        # Inject one new input connection for each configured label.
        for label in config.grouped_labels():
            self.inputs.add(label)
            dataset_type_name = f"{label}_metadata"
            self.allConnections[label] = ct.Input(
                dataset_type_name,
                storageClass="TaskMetadata",
                dimensions=config.dimensions,
                multiple=True,
                deferLoad=True,
            )
            # No actual name overrides permitted (these connections aren't
            # individually configurable, because they're dynamically added),
            # but for now we need to populate these internal things manually
            # because super().__init__ can only populate them with the usual
            # static connections.
            self._nameOverrides[label] = dataset_type_name
            self._typeNameToVarName[dataset_type_name] = label


class GatherResourceStatisticsConfig(
    PipelineTaskConfig,
    pipelineConnections=GatherResourceStatisticsConnections
):
    dimensions = ListField(
        "The quantum dimensions for all tasks whose resources are being "
        "aggregated, and the columns (after expansion to include implied "
        "dimensions) used to identify rows in the output table.",
        dtype="str",
    )
    labels = ListField(
        "Pipeline labels for the tasks whose resource usage should be "
        "gathered, or '{label}.{method}' identifiers for methods within "
        "tasks whose resource statistics should be separately gathered. "
        "The special names '{label}.init' and '{label}.prep' can be used "
        "to record the time spent in task-construction and checking for "
        "available input datasets (prior to task construction), respectively.",
        dtype="str",
    )

    def grouped_labels(self):
        """Group self.labels by task label

        Returns
        -------
        grouped : `defaultdict` [ `str`, `set` [ `str` ] ]
            Dictionary whose keys are actual task labels, and whose values
            are sets whose elements are method names, the empty string (to
            indicate the full task's resource usage), or the special values
            'init' and 'prep'.
        """
        grouped = defaultdict(set)
        for label in self.labels:
            first, _, second = label.partition(".")
            grouped[first].add(second)
        return grouped


class GatherResourceStatisticsTask(PipelineTask):
    ConfigClass = GatherResourceStatisticsConfig
    _DefaultName = "gatherResourceStatistics"

    def run(self, **kwargs):
        # kwargs should be a mapping from task label to a list of
        # DeferredDatasetHandle.  We want a nested mapping, keyed first by data
        # ID, then by task label, with DeferredDatasetHandles as innermost
        # values.
        handles_by_data_id_then_label = defaultdict(dict)
        for task_label, handles in kwargs.items():
            for handle in handles:
                handles_by_data_id_then_label[handle.dataId][task_label] = handle
        # Start a dict of columns that we'll ultimately make into a table,
        # with just the data ID columns for now.
        raise NotImplementedError("TODO")
