#!/usr/bin/env python

from abc import ABC, abstractmethod
import itertools

import panel
import param
import numpy as np
import pandas as pd
import bokeh


from lsst.daf.butler import Butler


def naive_compute_cost(cpu_time, max_memory, memory_per_core=4E9):
    cores_occupied = np.ceil(max_memory / memory_per_core)
    return cpu_time * cores_occupied


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


class ResourceUsageData:

    _OBSERVATION_DIMENSIONS = frozenset({"visit", "exposure"})

    def __init__(self, tables):
        self.all_tasks = pd.Index(sorted(tables.keys()), name="task")
        self._full_tables = {task: self._with_resource_columns(tables[task]) for task in self.all_tasks}
        self.summary = pd.DataFrame(
            {
                NaiveComputeCost.__name__: np.zeros(len(self.all_tasks), dtype=float),
                "quanta": np.zeros(len(self.all_tasks), dtype=int),
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
        self._task_mask = np.ones(len(self.all_tasks), dtype=bool)

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

    def get_filtered_table(self, task):
        return self._full_tables[task]

    def __iter__(self):
        for task in self.all_tasks:
            yield task, self.get_filtered_table(task)

    @property
    def task_mask(self):
        return self._task_mask

    @task_mask.setter
    def task_mask(self, mask):
        self._task_mask[:] = mask

    def plot_costs(self):
        data = self.summary[self.task_mask].sort_values(NaiveComputeCost.__name__)
        data["fraction"] = data[NaiveComputeCost.__name__] / data[NaiveComputeCost.__name__].sum()
        data["angle"] = 2.0*np.pi*data["fraction"]
        data["annotation"] = [
            f"{task}: {cost:0.2f} core × hr ({fraction*100:0.2f}%)"
            for task, cost, fraction in zip(data.index, data[NaiveComputeCost.__name__], data["fraction"])
        ]
        colors = np.empty(len(data), dtype=(str, 8))
        for n, color in zip(
            range(0, len(data)),
            itertools.cycle(bokeh.palettes.Category20[20])
        ):
            colors[n] = color
        data["color"] = colors

        figure = bokeh.plotting.figure(
            y_range=(-0.5, 0.5),
            x_range=(-0.5, 0.5),
            toolbar_location=None,
            tools="hover",
            tooltips="@annotation",
        )
        figure.wedge(
            x=0,
            y=0,
            radius=0.4,
            start_angle=bokeh.transform.cumsum('angle', include_zero=True),
            end_angle=bokeh.transform.cumsum('angle'),
            line_width=0.0,
            fill_color="color",
            source=data,
        )
        figure.axis.axis_label = None
        figure.axis.visible = False
        figure.grid.grid_line_color = None
        return figure


class Dashboard(param.Parameterized):
    data = param.Parameter()
    observation = param.Selector(
        label="Include tasks with visit or exposure dimensions:",
        objects={"Unconstrained": None, "Yes": True, "No": False},
        default=None,
    )
    tract = param.Selector(
        label="Include tasks with tract dimensions",
        objects={"Unconstrained": None, "Yes": True, "No": False},
        default=None,
    )

    def set_task_mask(self):
        criteria = [np.ones(len(self.data.summary), dtype=bool)]
        if self.observation:
            criteria.append(self.data.summary["per_observation"])
        elif self.observation is False:
            criteria.append(np.logical_not(self.data.summary["per_observation"]))
        if self.tract:
            criteria.append(self.data.summary["per_tract"])
        elif self.tract is False:
            criteria.append(np.logical_not(self.data.summary["per_tract"]))
        self.data.task_mask = np.logical_and.reduce(criteria)

    @param.depends("observation", "tract", on_init=True)
    def info(self):
        self.set_task_mask()
        quanta = self.data.summary["quanta"]
        cost = self.data.summary[NaiveComputeCost.__name__]
        return panel.pane.Markdown(
            f"{self.data.task_mask.sum()}/{len(self.data.summary)} tasks, "
            f"{quanta[self.data.task_mask].sum()}/{quanta.sum()} quanta, "
            f"{100*cost[self.data.task_mask].sum()/cost.sum():0.02f}% of total compute cost",
            sizing_mode="stretch_width",
        )

    @param.depends("observation", "tract", on_init=True)
    def plot_costs(self):
        self.set_task_mask()
        return self.data.plot_costs()

    def panel(self):
        return panel.Column(
            panel.Row(
                self.param.observation,
                self.param.tract,
                self.info,
            ),
            self.plot_costs,
        )


def main():
    butler = Butler("/home/jbosch/LSST/tkt/DM-33963/EXPORTED_DATA", collections="u/jbosch/DM-33963")
    tables = {
        ref.datasetType.name[:-len("_resource_usage")]: butler.getDirect(ref)
        for ref in butler.registry.queryDatasets("*_resource_usage", components=False)
    }
    data = ResourceUsageData(tables)
    dashboard = Dashboard(data=data)
    dashboard.panel().servable()


main()
