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
        self.all_tasks = np.array(sorted(tables.keys()))
        self.palette = palette
        self._full_tables = {task: self._with_resource_columns(tables[task]) for task in self.all_tasks}
        self.full_summary = pd.DataFrame(
            {
                "task": self.all_tasks,
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
        )
        self._rebuild_summary()
        self._sort_tasks_by = NaiveComputeCost.__name__
        self._sort_tasks_reverse = True
        self._task_mask = np.ones(len(self.all_tasks), dtype=bool)
        self._active_task_colors = dict(
            zip(
                self._compute_active_tasks(self._task_mask, self._sort_tasks_by, self._sort_tasks_reverse),
                itertools.cycle(self.palette),
            )
        )

    @classmethod
    def _with_resource_columns(cls, table):
        result = pd.DataFrame(table)
        for rs in RESOURCE_TYPES:
            result[rs.__name__] = rs()(table)
        return result

    def _rebuild_summary(self):
        self.full_summary["quanta"] = [len(self[task]) for task in self.all_tasks]
        self.full_summary[NaiveComputeCost.__name__] = [
            self[task][NaiveComputeCost.__name__].sum() for task in self.all_tasks
        ]
        self.full_summary[MaxMemory.__name__] = [
            self[task][MaxMemory.__name__].max() for task in self.all_tasks
        ]
        self.full_summary[Runtime.__name__] = [
            self[task][Runtime.__name__].max() for task in self.all_tasks
        ]

    def __getitem__(self, task):
        return self._full_tables[task]

    def _compute_active_tasks(self, task_mask, sort_by, reverse):
        sorter = self.full_summary[sort_by].to_numpy()[self.task_mask].argsort()
        if reverse:
            sorter = sorter[::-1]
        return self.full_summary["task"].to_numpy()[self.task_mask][sorter]

    def sort_tasks(self, by, recolor=False, reverse=None):
        active_tasks = self._compute_active_tasks(self._task_mask, by, reverse=reverse)
        if recolor:
            self._active_task_colors = dict(zip(active_tasks, itertools.cycle(self.palette)))
        else:
            self._active_task_colors = {task: self._active_task_colors[task] for task in active_tasks}
        self._sort_tasks_by = by
        self._sort_tasks_reverse = reverse

    @property
    def task_mask(self):
        return self._task_mask

    @task_mask.setter
    def task_mask(self, mask):
        active_tasks = self._compute_active_tasks(mask, self._sort_tasks_by, reverse=self._sort_tasks_reverse)
        self._active_task_colors = dict(zip(active_tasks, itertools.cycle(self.palette)))
        self._task_mask = mask

    @property
    def tasks(self):
        return self._active_task_colors.keys()

    @property
    def colors(self):
        return self._active_task_colors.values()

    def summary(self):
        result = self.full_summary[self.task_mask].sort_values(self._sort_tasks_by, ignore_index=True)
        if self._sort_tasks_reverse:
            result = result[::-1]
        return result

    def tables(self):
        for task in self.tasks:
            yield self[task]

    def plot_costs(self):
        data = self.summary()
        data["fraction"] = data[NaiveComputeCost.__name__] / data[NaiveComputeCost.__name__].sum()
        data["angle"] = 2.0*np.pi*data["fraction"]
        data["annotation"] = [
            f"{task}: {cost:0.2f} core × hr ({fraction*100:0.2f}%)"
            for task, cost, fraction in zip(data["task"], data[NaiveComputeCost.__name__], data["fraction"])
        ]
        data["color"] = self.colors

        self._cost_pie_data = bokeh.models.ColumnDataSource(data)

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
            fill_alpha=0.5,
            nonselection_line_width=0.0,
            selection_line_width=3.0,
            nonselection_alpha=0.5,
            selection_alpha=1.0,
            line_color="black",
            fill_color="color",
            source=self._cost_pie_data,
        )
        figure.axis.axis_label = None
        figure.axis.visible = False
        figure.grid.grid_line_color = None
        if False:
            figure.add_layout(
                bokeh.models.Legend(
                    items=[
                        bokeh.models.LegendItem(label=task, index=n, renderers=[renderer])
                        for n, task in enumerate(self.tasks)
                    ],
                    click_policy="mute",
                ),
                "right"
            )
        return figure

    def plot_hist(self, ResourceClass):
        figure = bokeh.plotting.figure()
        for _, color, table in zip(self):
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
    sort_key = param.Selector(
        label="Sort tasks by",
        objects={
            "Task Label": "task",
            "Total Compute Cost": NaiveComputeCost.__name__,
            "Maximum Memory": MaxMemory.__name__,
            "Maximum CPU Time": Runtime.__name__,
        },
        default=NaiveComputeCost.__name__,
    )
    sort_desc = param.Boolean(label="Descending", default=True)
    task_selector = param.ListSelector()

    @param.depends("observation", "tract", watch=True)
    def set_task_mask(self):
        criteria = [np.ones(len(self.data.full_summary), dtype=bool)]
        if self.observation:
            criteria.append(self.data.full_summary["per_observation"])
        elif self.observation is False:
            criteria.append(np.logical_not(self.data.full_summary["per_observation"]))
        if self.tract:
            criteria.append(self.data.full_summary["per_tract"])
        elif self.tract is False:
            criteria.append(np.logical_not(self.data.full_summary["per_tract"]))
        self.data.task_mask = np.logical_and.reduce(criteria)

    @param.depends("sort_key", "sort_desc", watch=True)
    def sort_tasks(self):
        self.data.sort_tasks(self.sort_key, reverse=self.sort_desc)

    @param.depends("set_task_mask", on_init=True)
    def info(self):
        quanta = self.data.full_summary["quanta"]
        cost = self.data.full_summary[NaiveComputeCost.__name__]
        return panel.pane.Markdown(
            f"{self.data.task_mask.sum()}/{len(self.data.full_summary)} tasks, "
            f"{quanta[self.data.task_mask].sum()}/{quanta.sum()} quanta, "
            f"{100*cost[self.data.task_mask].sum()/cost.sum():0.02f}% of total compute cost",
            sizing_mode="stretch_width",
        )

    @param.depends("set_task_mask", "sort_tasks", on_init=True)
    def plot_costs_pie(self):
        return self.data.plot_costs()

    @param.depends("set_task_mask", "sort_tasks", on_init=True, watch=True)
    def _update_task_selector(self):
        self.param.task_selector.objects = list(range(len(self.data.tasks)))

    def _debug(self, *events):
        print(self.param.task_selector.objects)
        self.data._cost_pie_data.selected.indices = self.task_selector

    def panel(self):
        ts = panel.Param(
            self.param.task_selector,
            widgets={
                "task_selector": dict(
                    sizing_mode="stretch_height",
                )
            },
        )
        button = panel.widgets.Button(name='Debug', button_type='primary')
        button.on_click(self._debug)
        layout = panel.Column(
            panel.Row(
                self.param.observation,
                self.param.tract,
                self.info,
            ),
            panel.Row(
                panel.Column(
                    self.param.sort_key,
                    self.param.sort_desc,
                    ts,
                    button,
                ),
                self.plot_costs_pie,
            )
        )
        ts.widget("task_selector").jslink(self.data._cost_pie_data, value="selected", bidirectional=True)
        return layout


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
