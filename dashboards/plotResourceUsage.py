#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import enum
import itertools
from typing import Callable, Iterable, Mapping

import panel
import param
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
        print(dimension, values)
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
    def labels(self) -> tuple[str, str]:
        raise NotImplementedError()

    @property
    def aggregators(self) -> tuple[str, ...]:
        return ("mean", "max")


class MaxMemory(ResourceDefinition):
    def __call__(self, table: pd.DataFrame) -> pd.Series:
        return table["memory"] / 1E6

    @property
    def labels(self):
        return "peak memory (MB)", "maximum resident set size (MB)"


class Runtime(ResourceDefinition):
    def __call__(self, table):
        return table["run_time"]

    @property
    def labels(self) -> tuple[str, str]:
        return "peak time (s)", "CPU time in runQuantum (s)"


class NaiveComputeCost(ResourceDefinition):
    def __call__(self, table: pd.DataFrame) -> pd.Series:
        return naive_compute_cost(table["run_time"], table["memory"])

    @property
    def labels(self) -> tuple[str, str]:
        return "total cost (core × hr)", "[CPU time (s)] × [# of required (4GB) cores, assuming no retries]"

    @property
    def aggregators(self) -> tuple[str, ...]:
        return ("mean", "max", "sum")


RESOURCE_TYPES: tuple[type[ResourceDefinition], ...] = [MaxMemory, Runtime, NaiveComputeCost]

DEFAULT_PALETTE: tuple[str, ...] = bokeh.palettes.Category10[10]

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
                if not all(table["visit_system"] == 0):
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
        return CostBreakdownChart(self.group_by(by_columns, resources={NaiveComputeCost: "sum"}))


class CostBreakdownChart:

    def __init__(self, table: pd.DataFrame, palette: tuple[str, ...] = DEFAULT_PALETTE):
        self.table = table
        print(len(table))
        self.table["color"] = np.array(
            [color for _, color in zip(range(len(self.table)), itertools.cycle(palette))]
        )
        self.table["fraction"] = (
            self.table[NaiveComputeCost.__name__] / self.table[NaiveComputeCost.__name__].sum()
        )
        self.table["angle"] = 2.0 * np.pi * self.table["fraction"]

    def plot(self):
        figure = bokeh.plotting.figure(
            y_range=(-0.5, 0.5),
            x_range=(-0.5, 0.5),
            toolbar_location=None,
            tools=("tap",),
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
            source=self.table,
        )
        figure.axis.axis_label = None
        figure.axis.visible = False
        figure.grid.grid_line_color = None
        return figure


class Dashboard(param.Parameterized):
    raw_data = param.Parameter()
    task_filtered_data = param.Parameter()
    dimension_filtered_data = param.Parameter()
    per_band = param.Selector(
        label="band",
        objects={c.value: c for c in TaskDimensionConstraint},
        default=TaskDimensionConstraint.EITHER,
    )
    per_tract = param.Selector(
        label="tract",
        objects={c.value: c for c in TaskDimensionConstraint},
        default=TaskDimensionConstraint.EITHER,
    )
    per_visit = param.Selector(
        label="visit",
        objects={c.value: c for c in TaskDimensionConstraint},
        default=TaskDimensionConstraint.NO,
    )
    group_by = param.ListSelector(
        label="group by",
        objects=["task"],
        default=[],
    )
    allowed_bands = param.ListSelector(
        label="values",
        default=[],
    )
    allowed_tracts = param.ListSelector(
        label="values",
        default=[],
    )
    allowed_visits = param.ListSelector(
        label="values",
        default=[],
    )

    def _update_allowed_dimension(self, dimension, field, sort_key=None):
        getattr(self, field).clear()
        if dimension in self.task_filtered_data.dimensions:
            values = sorted(
                set(self.task_filtered_data.table[dimension]),
                key=sort_key
            )
            getattr(self.param, field).objects = values
            getattr(self, field)[:] = values
        else:
            getattr(self.param, field).objects = []

    @param.depends("per_band", "per_tract", "per_visit", on_init=True, watch=True)
    def _update_task_filters(self) -> None:
        self.task_filtered_data = self.raw_data.filter_tasks(
            band=self.per_band,
            tract=self.per_tract,
            visit=self.per_visit,
        )
        old_group_by = set(self.group_by)
        self.param.group_by.objects = [
            column for column in list(GROUPING_DIMENSIONS.keys())
            if column in self.task_filtered_data.independent_columns
        ]
        self.group_by = list(old_group_by & self.task_filtered_data.independent_columns)
        self._update_allowed_dimension("band", "allowed_bands", sort_key=band_sorter)
        self._update_allowed_dimension("tract", "allowed_tracts")
        self._update_allowed_dimension("visit", "allowed_visits")

    @param.depends(
        "_update_task_filters",
        "allowed_bands",
        "allowed_tracts",
        "allowed_visits",
        on_init=True,
        watch=True,
    )
    def _update_dimension_filters(self) -> None:
        self.dimension_filtered_data = self.task_filtered_data.filter_dimensions(
            predicate=PredicateIntersection.build(
                DimensionValueInSet.build("band", self.allowed_bands),
                DimensionValueInSet.build("tract", self.allowed_tracts),
                DimensionValueInSet.build("visit", self.allowed_visits),
            )
        )

    @param.depends("dimension_filtered_data", "group_by", on_init=True)
    def table(self) -> pd.DataFrame:
        df = self.dimension_filtered_data.group_by(["task"] + self.group_by)
        return panel.pane.DataFrame(df, sizing_mode="stretch_both")

    @param.depends("dimension_filtered_data", on_init=True)
    def cost_breakdown_chart(self) -> bokeh.plotting.Figure:
        return self.dimension_filtered_data.cost_breakdown_chart(["task"] + self.group_by).plot()

    def panel(self):
        layout = panel.Column(
            panel.Row(
                panel.Column(
                    self.param.per_band,
                    customized(self.param.allowed_bands, size=8),
                ),
                panel.Column(
                    self.param.per_tract,
                    customized(self.param.allowed_tracts, size=8),
                ),
                panel.Column(
                    self.param.per_visit,
                    customized(self.param.allowed_visits, size=8),
                ),
                customized(self.param.group_by, sizing_mode="stretch_height"),
            ),
            self.cost_breakdown_chart,
            self.table,
        )
        return layout


def main():
    butler = Butler("/home/jbosch/LSST/tkt/DM-33963/EXPORTED_DATA")
    raw_data = RawResourceUsageData.from_butler_query(butler, collections="u/jbosch/DM-33963")
    dashboard = Dashboard(raw_data=raw_data)
    dashboard.panel().servable()


main()
