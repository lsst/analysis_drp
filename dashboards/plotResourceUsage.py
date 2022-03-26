#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import enum
from typing import Callable, Iterable, Mapping

import panel
import numpy as np
import pandas as pd
import bokeh.plotting
import bokeh.models
import bokeh.palettes


from lsst.daf.butler import Butler


def naive_compute_cost(cpu_time, max_memory, memory_per_core=4E9):
    cores_occupied = np.ceil(max_memory / memory_per_core)
    return cpu_time * cores_occupied


def accept_everything(table: pd.DataFrame) -> np.ndarray:
    return np.ones(len(table), dtype=bool)


class DimensionValueInSet:

    @classmethod
    def build(cls, dimension: str, values: Iterable | None) -> Callable[[pd.DataFrame], np.ndarray]:
        if not values:
            return accept_everything
        else:
            return cls(dimension, values)

    def __init__(self, dimension: str, values: Iterable):
        self.dimension = dimension
        self.values = frozenset(values)

    def __call__(self, table: pd.DataFrame) -> np.ndarray:
        return table[self.dimension].isin(self.values).to_numpy()


class PredicateIntersection:

    @classmethod
    def build(
        cls,
        *nested: Callable[[pd.DataFrame], np.ndarray],
    ) -> Callable[[pd.DataFrame], np.ndarray]:
        filtered = [c for c in nested if c is not accept_everything]
        if not filtered:
            return accept_everything
        elif len(filtered) == 1:
            return filtered[0]
        else:
            return cls(filtered)

    def __init__(self, nested: list[Callable[[pd.DataFrame], np.ndarray]]):
        assert len(nested) > 1
        self._nested = nested

    def __call__(self, table: pd.DataFrame):
        return np.logical_and.reduce([other(table) for other in self._nested])


def band_sorter(b):
    try:
        return "ugrizy".index(b)
    except ValueError:
        if b.startswith("N"):
            return 10000 + int(b[1:])


def customized(p, **kwargs):
    return panel.Param(
        p,
        widgets={
            p: {p.name: kwargs},
        }
    )


class ResourceDefinition(ABC):
    @abstractmethod
    def __call__(self, table: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    @property
    @abstractmethod
    def labels(self) -> tuple[str, str, str]:
        raise NotImplementedError()

    @property
    def aggregators(self) -> tuple[str, ...]:
        return ("mean", "max")


class MaxMemory(ResourceDefinition):
    def __call__(self, table: pd.DataFrame) -> pd.Series:
        return table["memory"] / 1E6

    @property
    def labels(self) -> tuple[str, str, str]:
        return "memory", "MB", "maximum resident set size"


class Runtime(ResourceDefinition):
    def __call__(self, table):
        return table["run_time"]

    @property
    def labels(self) -> tuple[str, str, str]:
        return "time", "s", "CPU time in runQuantum"


class NaiveComputeCost(ResourceDefinition):
    def __call__(self, table: pd.DataFrame) -> pd.Series:
        return naive_compute_cost(table["run_time"], table["memory"]) / 3600.0

    @property
    def labels(self) -> tuple[str, str, str]:
        return "cost", "core×hr", "[CPU time] × [# of required (4GB) cores, assuming no retries]"

    @property
    def aggregators(self) -> tuple[str, ...]:
        return ("mean", "max", "sum")


RESOURCE_TYPES: tuple[type[ResourceDefinition], ...] = [MaxMemory, Runtime, NaiveComputeCost]

DEFAULT_PALETTE: tuple[str, ...] = bokeh.palettes.Category20[20]


def invisible_figure(x_range=None, y_range=None):
    """Make an invisible Bokeh figure.

    This is a part of a workaround for the fact that Bokeh legends can only
    be added to figures, not layouts, lifted largely from

    https://stackoverflow.com/questions/56825350/how-to-add-one-legend-for-that-controlls-multiple-bokeh-figures
    """
    figure = bokeh.plotting.figure(toolbar_location=None, outline_line_alpha=0)
    for fig_component in [figure.grid[0], figure.ygrid[0], figure.xaxis[0], figure.yaxis[0]]:
        fig_component.visible = False
    if x_range is not None:
        figure.x_range.start = x_range[0]
        figure.x_range.end = x_range[1]
    if y_range is not None:
        figure.y_range.start = y_range[0]
        figure.y_range.end = y_range[1]
    return figure


# Keys are dimensions we might want to group by in plots.
# Values are sets of other dimensions we should "roll up" into these,
# but only when the dimensions don't share a direct dependency in the butler
# data model.  Classic case is visit <-> exposure, where we want to roll up
# the per-exposure 'isr' quanta resource usage into per-visit resource usage.
GROUPING_DIMENSIONS: dict[str, frozenset[str]] = {
    "band": set(),
    "tract": set(),
    "visit": {"exposure"},
}


GROUPING_COLUMNS = ("task",) + tuple(GROUPING_DIMENSIONS.keys())


class TaskDimensionConstraint(enum.Enum):
    EITHER = "task may have this dimension"
    YES = "task must have dimension"
    NO = "task must not have dimension"


@dataclass
class RawResourceUsageData:
    tables: dict[str, pd.DataFrame]
    tasks_with_dimensions: dict[str, set[str]]

    @classmethod
    def from_butler_query(cls, butler: Butler, **kwargs):
        tables = {}
        tasks_with_dimensions = defaultdict(set)
        for ref in butler.registry.queryDatasets(
            "*_resource_usage",
            **kwargs,
            components=False,
        ):
            table = butler.getDirect(ref)
            task = ref.datasetType.name[:-len("_resource_usage")]
            for dimension, rollup_dimensions in GROUPING_DIMENSIONS.items():
                if dimension in table.columns or not rollup_dimensions.isdisjoint(table.columns):
                    tasks_with_dimensions[dimension].add(task)
            table["task"] = task
            for rs in RESOURCE_TYPES:
                table[rs.__name__] = rs()(table)
            tables[task] = table
        return cls(tables=tables, tasks_with_dimensions=dict(tasks_with_dimensions))

    def filter_tasks(self, **kwargs: TaskDimensionConstraint) -> TaskFilteredResourceUsageData:
        dimensions = frozenset(
            {
                dimension for dimension, constraint in kwargs.items()
                if constraint is TaskDimensionConstraint.YES
            }
        )
        tasks = set(self.tables.keys())
        for dimension, constraint in kwargs.items():
            if constraint is TaskDimensionConstraint.YES:
                tasks &= self.tasks_with_dimensions[dimension]
            elif constraint is TaskDimensionConstraint.NO:
                tasks &= (self.tables.keys() - self.tasks_with_dimensions[dimension])
        to_concat = []
        for task in tasks:
            table = self.tables[task]
            columns = {"task": np.array([task] * len(table), dtype=object)}
            for dimension in dimensions:
                columns[dimension] = self._get_dimension_column(dimension, table)
            for rs in RESOURCE_TYPES:
                columns[rs.__name__] = table[rs.__name__]
            task_filtered_table = pd.DataFrame(columns)
            to_concat.append(task_filtered_table)
        return TaskFilteredResourceUsageData(
            raw=self,
            dimensions=dimensions,
            independent_columns=frozenset({"task"} | dimensions),
            table=pd.concat(to_concat),
        )

    def _get_dimension_column(self, dimension, table):
        if dimension in table.columns:
            return table[dimension]
        else:
            rollup_dimensions = GROUPING_DIMENSIONS[dimension]
            if dimension == "visit" and rollup_dimensions == {"exposure"}:
                if "visit_system" in table.columns and not all(table["visit_system"] == 0):
                    raise NotImplementedError(
                        "Cannot yet roll-up exposures into visits unless visit system is one-to-one."
                    )
                return table["exposure"]
            else:
                raise NotImplementedError(
                    f"Roll-up not implemented for {dimension} from {rollup_dimensions}."
                )


@dataclass
class TaskFilteredResourceUsageData:
    raw: RawResourceUsageData
    dimensions: frozenset[str]
    independent_columns: frozenset[str]
    table: pd.DataFrame

    def filter_dimensions(
        self,
        predicate: Callable[[pd.DataFrame], np.ndarray] = accept_everything,
    ) -> DimensionFilteredResourceUsageData:
        return DimensionFilteredResourceUsageData(
            upstream=self,
            table=self.table[predicate(self.table)]
        )


@dataclass
class DimensionFilteredResourceUsageData:
    upstream: TaskFilteredResourceUsageData
    table: pd.DataFrame

    @property
    def raw(self) -> RawResourceUsageData:
        return self.upstream.raw

    @property
    def dimensions(self) -> frozenset[str]:
        return self.upstream.dimensions

    @property
    def independent_columns(self) -> frozenset[str]:
        return self.upstream.independent_columns

    def group_by(
        self,
        by_columns: Iterable[str] = ("task",),
        resources: Iterable[ResourceDefinition] = RESOURCE_TYPES,
    ) -> pd.DataFrame:
        by_columns = frozenset(by_columns)
        if not isinstance(resources, Mapping):
            resources = {rs: rs().aggregators for rs in resources}
        if not by_columns:
            raise RuntimeError("No columns to group by.")
        if not self.independent_columns.issuperset(by_columns):
            raise ValueError(
                f"Cannot sort by column(s) {by_columns - self.independent_columns}."
            )
        return self.table.groupby(list(by_columns)).agg({rs.__name__: agg for rs, agg in resources.items()})

    def cost_breakdown_chart(self, by_columns: Iterable[str]):
        table = self.group_by(by_columns, resources={NaiveComputeCost: "sum"})
        table.sort_values(NaiveComputeCost.__name__, inplace=True, ascending=False)
        table.reset_index(inplace=True)
        return CostBreakdownChart(table)


class CostBreakdownChart:

    def __init__(self, table: pd.DataFrame, palette: tuple[str, ...] = DEFAULT_PALETTE):
        self.n_showable = min(len(palette) - 1, len(table))
        self.table = table
        self.table["fraction"] = (
            self.table[NaiveComputeCost.__name__] / self.table[NaiveComputeCost.__name__].sum()
        )
        self.table["angle"] = 2.0 * np.pi * self.table["fraction"]
        label_template_terms = []
        self.tooltips = []
        if "task" in self.table.columns:
            label_template_terms.append("{row[task]}")
            self.tooltips.append(("task", "@task"))
        for dimension in GROUPING_DIMENSIONS.keys():
            if dimension in self.table.columns:
                label_template_terms.append(f"{dimension}={{row[{dimension}]}}")
                self.tooltips.append((dimension, f"@{dimension}"))
        self.label_template = ", ".join(label_template_terms)
        _, units, _ = NaiveComputeCost().labels
        self.tooltips.append(("cost", f"@{NaiveComputeCost.__name__}{{0.00}} {units}"))
        self.tooltips.append(("fraction", "@fraction{0.000}"))
        color = np.empty(len(table), dtype=object)
        color[:self.n_showable] = palette[:self.n_showable]
        color[self.n_showable:] = palette[-1]
        self.table["color"] = color

    def plot(self):
        _, _, long_title = NaiveComputeCost().labels
        figure = bokeh.plotting.figure(
            title=f"Naive compute cost breakdown {long_title}",
            y_range=(-0.5, 0.5),
            x_range=(-0.5, 0.5),
            toolbar_location=None,
            tools=("tap", "hover"),
            tooltips=self.tooltips,
        )
        renderer = figure.wedge(
            x=0,
            y=0,
            radius=0.4,
            start_angle=bokeh.transform.cumsum('angle', include_zero=True),
            end_angle=bokeh.transform.cumsum('angle'),
            line_width=0.0,
            fill_alpha=0.8,
            nonselection_line_width=0.0,
            selection_line_width=3.0,
            nonselection_alpha=0.8,
            selection_alpha=1.0,
            line_color="white",
            fill_color="color",
            source=self.table,
        )
        items = [
            bokeh.models.LegendItem(
                label=self.label_template.format(row=row),
                renderers=[renderer],
                index=n,
            )
            for n, (_, row) in enumerate(self.table[:self.n_showable].iterrows())
        ]
        items.append(
            bokeh.models.LegendItem(label="(all others)", renderers=[renderer], index=self.n_showable)
        )
        legend = bokeh.models.Legend(items=items)
        figure.axis.axis_label = None
        figure.axis.visible = False
        figure.grid.grid_line_color = None
        legend_figure = invisible_figure(x_range=(1.0, 1.1))
        # The glyphs referred by the legend need to be present in the figure
        # that holds the legend, so we must add them to the figure renderers,
        # but also set the figure range so they don't appear.
        legend_figure.renderers.append(renderer)
        legend_figure.add_layout(legend, place="left")
        return figure, legend_figure


class Dashboard:

    def __init__(self, raw_data: RawResourceUsageData):
        self.per_band = panel.widgets.Select(
            name="band",
            options={c.value: c for c in TaskDimensionConstraint},
            value=TaskDimensionConstraint.EITHER,
        )
        self.per_tract = panel.widgets.Select(
            name="tract",
            options={c.value: c for c in TaskDimensionConstraint},
            value=TaskDimensionConstraint.EITHER,
        )
        self.per_visit = panel.widgets.Select(
            name="visit",
            options={c.value: c for c in TaskDimensionConstraint},
            value=TaskDimensionConstraint.EITHER,
        )
        self.group_by = panel.widgets.MultiSelect(
            name="group by",
            options=["task"],
            value=["task"],
        )
        self.allowed_bands = panel.widgets.MultiSelect(
            name="values",
            options=[],
            value=[]
        )
        self.allowed_tracts = panel.widgets.MultiSelect(
            name="values",
            options=[],
            value=[]
        )
        self.allowed_visits = panel.widgets.MultiSelect(
            name="values",
            options=[],
            value=[]
        )
        self.raw_data = raw_data
        self.pie_chart = panel.pane.Bokeh(sizing_mode="stretch_width")
        self.legend = panel.pane.Bokeh(sizing_mode="fixed")
        self.table = panel.pane.DataFrame(
            sizing_mode="stretch_both",
            float_format=lambda v: f"{v:0.02f}",
            justify="center",
        )
        self._update_task_filters()

    def panel(self):
        layout = panel.Column(
            panel.Row(
                panel.Column(
                    self.per_band,
                    self.allowed_bands,
                ),
                panel.Column(
                    self.per_tract,
                    self.allowed_tracts,
                ),
                panel.Column(
                    self.per_visit,
                    self.allowed_visits,
                ),
                self.group_by,
            ),
            panel.Row(self.pie_chart, self.legend),
            self.table,
        )
        self.link()
        return layout

    def link(self):
        self.per_band.param.watch(self._update_task_filters, ["value"], onlychanged=True)
        self.per_tract.param.watch(self._update_task_filters, ["value"], onlychanged=True)
        self.per_visit.param.watch(self._update_task_filters, ["value"], onlychanged=True)
        self.allowed_bands.param.watch(self._update_dimension_filters, ["value"], onlychanged=True)
        self.allowed_tracts.param.watch(self._update_dimension_filters, ["value"], onlychanged=True)
        self.allowed_visits.param.watch(self._update_dimension_filters, ["value"], onlychanged=True)
        self.group_by.param.watch(self._update_group_by, ["value"], onlychanged=True)

    def _update_task_filters(self, *args) -> None:
        self.task_filtered_data = self.raw_data.filter_tasks(
            band=self.per_band.value,
            tract=self.per_tract.value,
            visit=self.per_visit.value,
        )
        old_group_by = set(self.group_by.value)
        self.group_by.options = ["task"] + [
            column for column in list(GROUPING_DIMENSIONS.keys())
            if column in self.task_filtered_data.independent_columns
        ]
        self.group_by.value = list(old_group_by & self.task_filtered_data.independent_columns)
        self._update_allowed_dimension("band", self.allowed_bands, sort_key=band_sorter)
        self._update_allowed_dimension("tract", self.allowed_tracts)
        self._update_allowed_dimension("visit", self.allowed_visits)
        self._update_dimension_filters()

    def _update_dimension_filters(self, *args) -> None:
        self.dimension_filtered_data = self.task_filtered_data.filter_dimensions(
            predicate=PredicateIntersection.build(
                DimensionValueInSet.build("band", self.allowed_bands.value),
                DimensionValueInSet.build("tract", self.allowed_tracts.value),
                DimensionValueInSet.build("visit", self.allowed_visits.value),
            )
        )
        self._update_panes()

    def _update_group_by(self, *args) -> None:
        if not self.group_by.value:
            self.group_by.value = ["task"]
        else:
            self.group_by.value.sort(key=GROUPING_COLUMNS.index)
        self._update_panes()

    def _update_panes(self, *args) -> None:
        self.pie_chart.object, self.legend.object = self.dimension_filtered_data.cost_breakdown_chart(
            self.group_by.value
        ).plot()
        self.table.object = self.dimension_filtered_data.group_by(self.group_by.value).rename(
            {rs.__name__: f"{rs().labels[0]} ({rs().labels[1]})" for rs in RESOURCE_TYPES},
            axis=1,
            level=0,
        )

    def _update_allowed_dimension(self, dimension, widget, sort_key=None):
        widget.value.clear()
        if dimension in self.task_filtered_data.dimensions:
            options = sorted(
                set(self.task_filtered_data.table[dimension]),
                key=sort_key
            )
            widget.options = options
            widget.value[:] = options
        else:
            widget.options = []


def main():
    butler = Butler("/home/jbosch/LSST/tkt/DM-33963/EXPORTED_DATA")
    raw_data = RawResourceUsageData.from_butler_query(butler, collections="u/jbosch/DM-33963")
    dashboard = Dashboard(raw_data=raw_data)
    dashboard.panel().servable()


main()
