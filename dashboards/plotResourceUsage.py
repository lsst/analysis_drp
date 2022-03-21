#!/usr/bin/env python

from abc import ABC, abstractmethod
import itertools

import panel
import param
import numpy as np
import pandas as pd
import bokeh


from lsst.daf.butler import Butler
BUTLER = Butler("/home/jbosch/LSST/tkt/DM-33963/EXPORTED_DATA", collections="u/jbosch/DM-33963")


def naive_compute_cost(cpu_time, max_memory, memory_per_core=4E9):
    cores_occupied = np.ceil(max_memory / memory_per_core)
    return cpu_time * cores_occupied


TABLES = {
    ref.datasetType.name[:-len("_resource_usage")]: BUTLER.getDirect(ref)
    for ref in BUTLER.registry.queryDatasets("*_resource_usage", components=False)
}


class ResourceDefinition(ABC):
    @abstractmethod
    def __call__(self, table):
        raise NotImplementedError()

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError()

    @abstractmethod
    def summarize(self, values):
        raise NotImplementedError()


class MaxMemory(ResourceDefinition):
    def __call__(self, table):
        return table["memory"] / 1E6

    @property
    def labels(self):
        return "peak memory (MB)", "maximum resident set size (MB)"

    def summarize(self, values):
        return values.max()


class Runtime(ResourceDefinition):
    def __call__(self, table):
        return table["run_time"]

    @property
    def labels(self):
        return "peak time (s)", "CPU time in runQuantum (s)"

    def summarize(self, values):
        return values.max()


class NaiveComputeCost(ResourceDefinition):
    def __call__(self, table):
        return naive_compute_cost(table["run_time"], table["memory"])

    @property
    def labels(self):
        return "total cost (core × hr)", "[CPU time (s)] × [# of required (4GB) cores, assuming no retries]"

    def summarize(self, values):
        return values.sum() / 3600.0


RESOURCE_TYPES = [MaxMemory, Runtime, NaiveComputeCost]


class TaskSelector(param.Parameterized):
    task_summary = param.Parameter()
    observation = param.Selector(
        label="Observation Dimensions",
        objects={"Unconstrained": None, "Yes": True, "No": False},
        default=None,
    )
    tract = param.Selector(
        label="Tract Dimensions",
        objects={"Unconstrained": None, "Yes": True, "No": False},
        default=None,
    )
    selector = param.ListSelector(objects=[], default=[])

    @param.depends("observation", "tract", on_init=True, watch=True)
    def _update_selector(self):
        criteria = [np.ones(len(self.task_summary), dtype=bool)]
        if self.observation:
            criteria.append(self.task_summary["per_observation"])
        elif self.observation is False:
            criteria.append(np.logical_not(self.task_summary["per_observation"]))
        if self.tract:
            criteria.append(self.task_summary["per_tract"])
        elif self.tract is False:
            criteria.append(np.logical_not(self.task_summary["per_tract"]))
        indexer = np.logical_and.reduce(criteria)
        self.param.selector.objects = list(self.task_summary.index[indexer])

    @param.depends("observation", "tract", on_init=True)
    def info(self):
        criteria = [np.ones(len(self.task_summary), dtype=bool)]
        if self.observation:
            criteria.append(self.task_summary["per_observation"])
        elif self.observation is False:
            criteria.append(np.logical_not(self.task_summary["per_observation"]))
        if self.tract:
            criteria.append(self.task_summary["per_tract"])
        elif self.tract is False:
            criteria.append(np.logical_not(self.task_summary["per_tract"]))
        indexer = np.logical_and.reduce(criteria)
        quanta = self.task_summary["quanta"]
        cost = self.task_summary[NaiveComputeCost.__name__]
        return panel.pane.Markdown(
            f"{indexer.sum()}/{len(self.task_summary)} tasks, "
            f"{quanta[indexer].sum()}/{quanta.sum()} quanta, "
            f"{100*cost[indexer].sum()/cost.sum():0.02f}% of total compute cost",
            sizing_mode="stretch_width",
        )

    def panel(self):
        return panel.Column(
            panel.Row(
                self.param.observation,
                self.param.tract,
                self.info,
            ),
            panel.Param(self.param.selector, widgets={"selector": dict(sizing_mode="stretch_height")}),
        )


class ResourceUsageData:

    _OBSERVATION_DIMENSIONS = frozenset({"visit", "exposure"})

    def __init__(self, tables):
        self.all_tasks = pd.Index(sorted(tables.keys()), name="task")
        self._full_tables = {task: self._with_resource_columns(tables[task]) for task in self.all_tasks}
        self.summary = pd.DataFrame(
            {
                NaiveComputeCost.__name__: np.zeros(len(self.all_tasks), dtype=float),
                "quanta": np.zeros(len(self.all_tasks), dtype=int),
                "color": np.array(["#ffffff"] * len(self.all_tasks), dtype=(str, 8)),
                "enabled": np.zeros(len(self.all_tasks), dtype=bool),
                "per_observation": np.array(
                    [
                        not self._OBSERVATION_DIMENSIONS.isdisjoint(self._full_tables[task].columns)
                        for task in self.all_tasks
                    ],
                    dtype=bool,
                ),
                "per_tract": np.array(
                    [
                        "tract" in self._full_tables[task].columns
                        for task in self.all_tasks
                    ],
                    dtype=bool,
                ),
            },
            index=self.all_tasks,
        )
        self._rebuild_summary()
        self.task_selector = TaskSelector(task_summary=self.summary)

    @classmethod
    def _with_resource_columns(cls, table):
        result = pd.DataFrame(table)
        for rs in RESOURCE_TYPES:
            result[rs.__name__] = rs()(table)
        return result

    def _rebuild_summary(self):
        self.summary["quanta"] = [len(table) for _, table in self]
        self.summary[NaiveComputeCost.__name__] = [
            table[NaiveComputeCost.__name__].sum() for _, table in self
        ]
        self.cost_sorter = np.argsort(self.summary[NaiveComputeCost.__name__])[::-1]

    def _initialize_display_columns(self):
        # Assign colors by rank in NaiveComputeCost total, and show the first
        # 20 tasks in that ranking
        for i, n, color in zip(
            itertools.count(0),
            self.cost_sorter,
            itertools.cycle(bokeh.palettes.Category20[20])
        ):
            self.summary["color"][n] = color
            self.summary["enabled"][n] = i < 20

    def get_filtered_table(self, task):
        return self._full_tables[task]

    def __iter__(self):
        for task in self.all_tasks:
            yield task, self.get_filtered_table(task)

    def panel(self):
        return self.task_selector.panel()


DATA = ResourceUsageData(TABLES)
DATA.panel().servable()
