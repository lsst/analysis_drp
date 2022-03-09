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

__all__ = (
    "GatherResourceStatisticsConfig",
    "GatherResourceStatisticsConnections",
    "GatherResourceStatisticsTask",
)

import numpy as np
import pandas as pd

from lsst.pex.config import Config, ConfigDictField, Field, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as ct

# It's not great to be importing a private symbol, but this is a temporary
# workaround for the fact that prior to w.2022.10, the units for memory values
# written in task metadata were platform-dependent.  Once we no longer care
# about older runs, this import and the code that uses it can be removed.
from lsst.utils.usage import _RUSAGE_MEMORY_MULTIPLIER


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
        ),
    )

    def __init__(self, *, config):
        super().__init__(config=config)
        # Inject one new input connection for each configured label.
        for label in config.labels:
            self.inputs.add(label)
            dataset_type_name = f"{label}_metadata"
            connection = ct.Input(
                dataset_type_name,
                storageClass="TaskMetadata",
                dimensions=config.dimensions,
                multiple=True,
                deferLoad=True,
            )
            setattr(self, label, connection)
            self.allConnections[label] = connection
            # No actual name overrides permitted (these connections aren't
            # individually configurable, because they're dynamically added),
            # but for now we need to populate these internal things manually
            # because super().__init__ can only populate them with the usual
            # static connections.
            self._nameOverrides[label] = dataset_type_name
            self._typeNameToVarName[dataset_type_name] = label


class ExtractResourceUsageConfig(Config):
    memory = Field(
        doc=(
            "Whether to extract peak memory usage (maximum resident set size) "
            "for this task. "
            "Note that memory usage cannot be further subdivided because only "
            "a per-process peak is available (and hence if multiple quanta "
            "are run in one quantum, even per-quantum values may be "
            "misleading)."
        ),
        dtype=bool,
        default=True,
    )
    prep_time = Field(
        doc=(
            "Whether to extract the CPU time duration for the work the "
            "middleware does prior to initializing the task (mostly checking "
            "for input dataset existence)."
        ),
        dtype=bool,
        default=False,
    )
    init_time = Field(
        doc=(
            "Whether to extract the CPU time duration for actually "
            "constructing the task."
        ),
        dtype=bool,
        default=True,
    )
    run_time = Field(
        doc=(
            "Whether to extract the CPU time duration for actually "
            "executing the task."
        ),
        dtype=bool,
        default=True,
    )
    method_times = ListField(
        doc=(
            "Names of @lsst.utils.timer.timeMethod-decorated methods for "
            "which CPU time durations should also be extracted.  Use '.' "
            "separators to refer to subtask methods at arbitrary depth."
        ),
        dtype=str,
        optional=False,
        default=[],
    )


class GatherResourceStatisticsConfig(
    PipelineTaskConfig, pipelineConnections=GatherResourceStatisticsConnections
):
    dimensions = ListField(
        doc=(
            "The quantum dimensions for all tasks whose resources are being "
            "aggregated, and the columns (after expansion to include implied "
            "dimensions) used to identify rows in the output table."
        ),
        dtype=str,
    )
    labels = ConfigDictField(
        doc=(
            "A mapping from task-label keys to configuration for which "
            "resource-usage quantities to gather from each task's metadata."
        ),
        keytype=str,
        itemtype=ExtractResourceUsageConfig,
        optional=False,
        default={},
    )


class GatherResourceStatisticsTask(PipelineTask):
    """A `PipelineTask` that gathers resource usage statistics from task
    metadata.

    Notes
    -----
    This is an unusual `PipelineTask` in that its input connections are
    dynamic: they are all of the metadata datasets produced by all of the tasks
    in `GatherResourceStatisticsConfig.labels`.

    Its output table has columns for each of the dimensions in
    `GatherResourceStatisticsConfig.labels` (as well as dimensions implied by
    those), to record the data ID, as well as (subject to configuration):

    - ``{label}.memory``: the maximum resident set size for the entire quantum
      (in bytes);
    - ``{label}.prep_time``: the time spent in the pre-initialization step in
      which the middleware checks which of the quantum's inputs are available;
    - ``{label}.init_time``: the time spent in task construction;
    - ``{label}.run_time``: the time spent executing the task's runQuantum
      method.
    - ``{label}.{method}``: the time spent in a particular task or subtask
      method decorated with `lsst.utils.timer.timeMethod`.

    All time durations are CPU times in seconds, and all columns are 64-bit
    floating point.  Methods or steps that did not run are given a duration of
    zero.

    It is expected that this task will be configured to run multiple times in
    most pipelines, with different output labels and output dataset type names,
    for each of the sets of dimensions whose tasks are prominent enough to be
    worth aggregating resource usage statistics.
    """

    ConfigClass = GatherResourceStatisticsConfig
    _DefaultName = "gatherResourceStatistics"

    def runQuantum(
        self,
        butlerQC,
        inputRefs,
        outputRefs,
    ) -> None:
        # Docstring inherited.
        # This override exists just so we can pass the butler registry's
        # DimensionUniverse to run in order to standardize the dimensions.
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(butlerQC.registry.dimensions, **inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, universe, **kwargs):
        """Gather resource usage statistics from per-quantum metadata.

        Parameters
        ----------
        universe : `DimensionUniverse`
            Object managing all dimensions recognized by the butler; used to
            standardize and expand `GatherResourceStatisticsConfig.dimensions`.
        **kwargs
            Keyword arguments map the configured task labels
            (keys of `GatherResourceStatisticsConfig.labels`) to lists of
            `lsst.daf.butler.DeferredDatasetHandle` that can be used to load
            all metadata datasets for that task.

        Returns
        -------
        result : `Struct`
            Structure with a single element:

            - ``outout_table``: a `pandas.DataFrame` that aggregates the
              configured resource usage statistics.
        """
        dimensions = universe.extract(self.config.dimensions)
        # kwargs should be a mapping from task label to a list of
        # DeferredDatasetHandle.  We want a nested mapping, keyed first by data
        # ID, then by task label, with DeferredDatasetHandles as innermost
        # values.
        handles_by_data_id_then_label = defaultdict(dict)
        for task_label, handles in kwargs.items():
            for handle in handles:
                handles_by_data_id_then_label[handle.dataId][task_label] = handle
        n_rows = len(handles_by_data_id_then_label)
        # Create a dict of empty column arrays that we'll ultimately make into
        # a table.
        columns = {
            d.name: np.zeros(n_rows, dtype=_dtype_from_field_spec(d.primaryKey))
            for d in dimensions
        }
        for task_label, subconfig in self.config.labels.items():
            for attr_name in ("memory", "prep_time", "init_time", "run_time"):
                if getattr(subconfig, attr_name):
                    columns[f"{task_label}.{attr_name}"] = np.zeros(n_rows, dtype=float)
            for method_name in subconfig.method_times:
                columns[f"{task_label}.{method_name}"] = np.zeros(n_rows, dtype=float)
        # Populate the table, one row at a time.
        warned_about_metadata_version = False
        for index, (data_id, handles) in enumerate(
            handles_by_data_id_then_label.items()
        ):
            # Fill in the data ID columns.
            for k, v in data_id.full.byName().items():
                columns[k][index] = v
            # Load the metadata datasets and fill in the columns derived from
            # them.
            for task_label, handle in handles.items():
                metadata = handle.get()
                subconfig = self.config.labels[task_label]
                try:
                    quantum_metadata = metadata["quantum"]
                except KeyError:
                    self.log.warning(
                        "Metadata for %s @ %s has no 'quantum' key.",
                        task_label,
                        handle.dataId,
                    )
                else:
                    # If the `quantum` metadata key exists, the `prep` and
                    # `end` prefix entries should always be present, so we
                    # don't do any extra exception handling for those.  The
                    # `init` and `start` prefix entries will only exist if the
                    # `prep` step determined that there was work to do, so
                    # those fall back to the end time to make the related
                    # durations zero.
                    if subconfig.memory:
                        # Attempt to work around memory units being
                        # platform-dependent for metadata written prior to
                        # w.2022.10.
                        memory_multiplier = 1
                        if quantum_metadata.get("__version__", 0) < 1:
                            memory_multiplier = _RUSAGE_MEMORY_MULTIPLIER
                            msg = (
                                "Metadata for %s @ %s is too old; guessing memory units by "
                                "assuming the platform has not changed"
                            )
                            if not warned_about_metadata_version:
                                self.log.warning(msg, task_label, handle.dataId)
                                self.log.warning(
                                    "Warnings about memory units for other inputs "
                                    "will be emitted only at DEBUG level."
                                )
                                warned_about_metadata_version = True
                            else:
                                self.log.debug(msg, task_label, handle.dataId)
                        columns[f"{task_label}.memory"][index] = (
                            quantum_metadata["endMaxResidentSetSize"]
                            * memory_multiplier
                        )
                    end_time = quantum_metadata["endCpuTime"]
                    times = [
                        quantum_metadata["prepCpuTime"],
                        quantum_metadata.get("initCpuTime", end_time),
                        quantum_metadata.get("startCpuTime", end_time),
                        end_time,
                    ]
                    for attr_name, begin, end in zip(
                        ["prep_time", "init_time", "run_time"],
                        times[:-1],
                        times[1:],
                    ):
                        if getattr(subconfig, attr_name):
                            columns[f"{task_label}.{attr_name}"][index] = end - begin
                for method_name in subconfig.method_times:
                    terms = [task_label] + list(method_name.split("."))
                    metadata_method_name = ":".join(terms[:-1]) + "." + terms[-1]
                    try:
                        method_start_time = metadata[
                            f"{metadata_method_name}StartCpuTime"
                        ]
                        method_end_time = metadata[f"{metadata_method_name}EndCpuTime"]
                    except KeyError:
                        # A method missing from the metadata is not a problem;
                        # it's reasonable for configuration or even runtime
                        # logic to result in a method not being called.  When
                        # that happens, we just let the times stay zero.
                        pass
                    else:
                        columns[f"{task_label}.{method_name}"] = (
                            method_end_time - method_start_time
                        )
        return Struct(output_table=pd.DataFrame(columns, copy=False))


def _dtype_from_field_spec(field_spec):
    """Return the `np.dtype` that can be used to hold the values of a butler
    dimension field.

    Parameters
    ----------
    field_spec : `lsst.daf.butler.core.ddl.FieldSpec`
        Object describing the field in a SQL-friendly sense.

    Returns
    -------
    dtype : `np.dtype`
        Numpy data type description.
    """
    python_type = field_spec.getPythonType()
    if python_type is str:
        return np.dtype((str, field_spec.length))
    else:
        return np.dtype(python_type)
