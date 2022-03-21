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

    def __init__(self, tables, palette=bokeh.palettes.Category10[10]):
        self.all_tasks = pd.Index(sorted(tables.keys()), name="task")
        self.palette = palette
        self._full_tables = {task: self._with_resource_columns(tables[task]) for task in self.all_tasks}
        self.summary = pd.DataFrame(
            {
                NaiveComputeCost.__name__: np.zeros(len(self.all_tasks), dtype=float),
                MaxMemory.__name__: np.zeros(len(self.all_tasks), dtype=float),
                Runtime.__name__: np.zeros(len(self.all_tasks), dtype=float),
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
        self._task_indices = np.arange(len(self.all_tasks), dtype=int)

    @classmethod
    def _with_resource_columns(cls, table):
        result = pd.DataFrame(table)
        for rs in RESOURCE_TYPES:
            result[rs.__name__] = rs()(table)
        return result

    def _rebuild_summary(self):
        self.summary["quanta"] = [len(table) for _, table in self.iter_tables(mask_tasks=False)]
        self.summary[NaiveComputeCost.__name__] = [
            table[NaiveComputeCost.__name__].sum() for _, table in self.iter_tables(mask_tasks=False)
        ]
        self.summary[MaxMemory.__name__] = [
            table[MaxMemory.__name__].max() for _, table in self.iter_tables(mask_tasks=False)
        ]
        self.summary[Runtime.__name__] = [
            table[Runtime.__name__].max() for _, table in self.iter_tables(mask_tasks=False)
        ]

    def get_filtered_table(self, task):
        return self._full_tables[task]

    def iter_tables(self, mask_tasks=True, sort_by=None):
        if not mask_tasks:
            task_mask = slice(None)
        else:
            task_mask = self.task_mask
        if sort_by is None:
            tasks = self.all_tasks[task_mask]
        else:
            tasks = self.summary[sort_by][task_mask].sort_values().index
        for task in tasks:
            yield task, self.get_filtered_table(task)

    def set_tasks(self, mask, sort_by, ):
        self._task_mask[:] = mask

    def plot_costs(self, sort_by):
        data = self.summary[self.task_mask].sort_values(sort_by)
        data["fraction"] = data[NaiveComputeCost.__name__] / data[NaiveComputeCost.__name__].sum()
        data["angle"] = 2.0*np.pi*data["fraction"]
        data["annotation"] = [
            f"{task}: {cost:0.2f} core × hr ({fraction*100:0.2f}%)"
            for task, cost, fraction in zip(data.index, data[NaiveComputeCost.__name__], data["fraction"])
        ]
        colors = np.empty(len(data), dtype=(str, 8))
        for n, color in zip(
            range(0, len(data)),
            itertools.cycle(self.palette)
        ):
            colors[n] = color
        data["color"] = colors

        figure = bokeh.plotting.figure(
            y_range=(-0.5, 0.5),
            x_range=(-0.5, 0.5),
            toolbar_location=None,
            tools=("hover", "tap"),
            tooltips="@annotation",
        )
        figure.wedge(
            x=0,
            y=0,
            radius=0.4,
            start_angle=bokeh.transform.cumsum('angle', include_zero=True),
            end_angle=bokeh.transform.cumsum('angle'),
            line_width=0.0,
            alpha=0.5,
            nonselection_line_width=0.0,
            selection_line_width=3.0,
            nonselection_alpha=0.5,
            selection_alpha=1.0,
            line_color="black",
            fill_color="color",
            source=data,
            legend_group="task",
        )
        figure.axis.axis_label = None
        figure.axis.visible = False
        figure.grid.grid_line_color = None
        return figure

    def plot_hist(self, sort_by, ResourceClass):
        figure = bokeh.plotting.figure()
        for (task, table), color in zip(self.iter_tables(sort_by=sort_by), itertools.cycle(self.palette)):
            if len(table) == 1:
                continue
            values = table[ResourceClass.__name__]
            hist, edges = np.histogram(values, density=True, bins=int(np.ceil(np.sqrt(len(table)))))
            figure.quad(
                left=edges[:-1],
                right=edges[1:],
                top=hist,
                bottom=0,
                line_color=color,
                fill_color=None,
                # legend_label=task,
            )
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
    sorter = param.Selector(
        label="Sort tasks by",
        objects={
            "Task Label": "task",
            "Total Compute Cost": NaiveComputeCost.__name__,
            "Maximum Memory": MaxMemory.__name__,
            "Maximum CPU Time": Runtime.__name__,
        },
        default=NaiveComputeCost.__name__,
    )

    @param.depends("observation", "tract", on_init=True, watch=True)
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

    @param.depends("set_task_mask", on_init=True)
    def info(self):
        quanta = self.data.summary["quanta"]
        cost = self.data.summary[NaiveComputeCost.__name__]
        return panel.pane.Markdown(
            f"{self.data.task_mask.sum()}/{len(self.data.summary)} tasks, "
            f"{quanta[self.data.task_mask].sum()}/{quanta.sum()} quanta, "
            f"{100*cost[self.data.task_mask].sum()/cost.sum():0.02f}% of total compute cost",
            sizing_mode="stretch_width",
        )

    @param.depends("set_task_mask", "sorter", on_init=True)
    def plot_costs_pie(self):
        return self.data.plot_costs(self.sorter)

    @param.depends("set_task_mask", "sorter", on_init=True)
    def plot_costs_hist(self):
        return self.data.plot_hist(self.sorter, NaiveComputeCost)

    def panel(self):
        return panel.Column(
            panel.Row(
                self.param.observation,
                self.param.tract,
                self.info,
            ),
            panel.Row(
                self.param.sorter,
                self.plot_costs_pie,
            )
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
