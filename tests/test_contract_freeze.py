from __future__ import annotations

import dataclasses
from enum import Enum
import inspect
import types
from typing import Any, get_args, get_origin, get_type_hints
import pytest

from catalog.catalog import Catalog
from catalog.stats import ColumnStats
import frontend
from frontend.binder import parse_and_bind
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, PlanNode, Project, Scan, Sort


PLAN_NODE_CLASSES = [Scan, Filter, Project, Join, Aggregate, Sort, Limit]
EXPR_CLASSES = [ColumnRef, Literal, BinaryOp, UnaryOp, AggCall]


def test_plannode_subclasses_exist() -> None:
    """Verifies all mandatory PlanNode subclasses exist and inherit from PlanNode."""
    for cls in PLAN_NODE_CLASSES:
        assert issubclass(cls, PlanNode), f"{cls.__name__} must inherit from PlanNode"


def test_plannode_frozen_dataclasses() -> None:
    """Verifies that all PlanNode subclasses are immutable (frozen) dataclasses."""
    for cls in PLAN_NODE_CLASSES:
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        params = getattr(cls, "__dataclass_params__", None)
        assert params is not None and params.frozen, f"{cls.__name__} must be a frozen dataclass"


def test_plannode_children_contract() -> None:
    """Verifies every PlanNode exposes a children property returning tuple[PlanNode, ...]."""
    for cls in PLAN_NODE_CLASSES:
        assert hasattr(cls, "children"), f"{cls.__name__} must have a 'children' attribute"
        static_attr = inspect.getattr_static(cls, "children")
        assert isinstance(static_attr, property), f"{cls.__name__}.children must be a property"
        hints = get_type_hints(static_attr.fget)
        ret_hint = hints.get("return")
        assert ret_hint is not None, f"{cls.__name__}.children getter must specify return type"
        assert get_origin(ret_hint) is tuple, f"{cls.__name__}.children must return a tuple"
        args = get_args(ret_hint)
        assert args[0] is PlanNode, f"{cls.__name__}.children elements must be PlanNode"


def test_plannode_schema_method_contract() -> None:
    """Verifies every PlanNode exposes schema() returning list[tuple[str, DType]]."""
    for cls in PLAN_NODE_CLASSES:
        assert hasattr(cls, "schema"), f"{cls.__name__} must have a 'schema' method"
        method = getattr(cls, "schema")
        assert callable(method), f"{cls.__name__}.schema must be callable"
        hints = get_type_hints(method)
        ret_hint = hints.get("return")
        assert ret_hint is not None, f"{cls.__name__}.schema must specify return type"
        assert get_origin(ret_hint) is list, f"{cls.__name__}.schema must return a list"
        elem_type = get_args(ret_hint)[0]
        assert get_origin(elem_type) is tuple, f"{cls.__name__}.schema elements must be tuples"
        tuple_args = get_args(elem_type)
        assert tuple_args == (str, DType), f"{cls.__name__}.schema tuple elements must be (str, DType)"


def test_plannode_replace_children_method_contract() -> None:
    """Verifies every PlanNode exposes replace_children(new_children) returning a PlanNode."""
    for cls in PLAN_NODE_CLASSES:
        assert hasattr(cls, "replace_children"), f"{cls.__name__} must have replace_children method"
        method = getattr(cls, "replace_children")
        assert callable(method), f"{cls.__name__}.replace_children must be callable"

        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        assert len(params) == 2, f"{cls.__name__}.replace_children must accept (self, new_children)"
        assert params[1].name == "new_children", f"Parameter must be named 'new_children' in {cls.__name__}"

        hints = get_type_hints(method)
        ret_hint = hints.get("return")
        assert ret_hint is not None, f"{cls.__name__}.replace_children must specify return type"
        assert issubclass(ret_hint, PlanNode), f"{cls.__name__}.replace_children return type must be PlanNode subclass"


def test_public_list_annotations_remain_strictly_lists() -> None:
    """Verifies public list annotations remain strictly Python lists (not sequences or tuples).

    Checked contracts:
    - Scan.columns: list[str] | None
    - Project.exprs: list[tuple[Expr, str]]
    - Aggregate.group_keys: list[Expr]
    - Aggregate.aggs: list[tuple[AggCall, str]]
    - Sort.keys: list[tuple[Expr, bool]]
    """
    # 1. Scan.columns: list[str] | None
    scan_hints = get_type_hints(Scan)
    scan_cols = scan_hints["columns"]
    # Union of list[str] and NoneType
    assert get_origin(scan_cols) in (types.UnionType, getattr(types, "Union", None))
    union_args = get_args(scan_cols)
    assert type(None) in union_args, "Scan.columns must allow None"
    list_arg = [a for a in union_args if a is not type(None)][0]
    assert get_origin(list_arg) is list, "Scan.columns non-null type must be strictly list"
    assert get_args(list_arg) == (str,), "Scan.columns item type must be str"

    # 2. Project.exprs: list[tuple[Expr, str]]
    proj_hints = get_type_hints(Project)
    proj_exprs = proj_hints["exprs"]
    assert get_origin(proj_exprs) is list, "Project.exprs must be strictly list"
    proj_tuple = get_args(proj_exprs)[0]
    assert get_origin(proj_tuple) is tuple, "Project.exprs elements must be tuples"
    assert get_args(proj_tuple) == (Expr, str), "Project.exprs elements must be tuple[Expr, str]"

    # 3. Aggregate.group_keys: list[Expr]
    agg_hints = get_type_hints(Aggregate)
    agg_gkeys = agg_hints["group_keys"]
    assert get_origin(agg_gkeys) is list, "Aggregate.group_keys must be strictly list"
    assert get_args(agg_gkeys) == (Expr,), "Aggregate.group_keys elements must be Expr"

    # 4. Aggregate.aggs: list[tuple[AggCall, str]]
    agg_aggs = agg_hints["aggs"]
    assert get_origin(agg_aggs) is list, "Aggregate.aggs must be strictly list"
    agg_tuple = get_args(agg_aggs)[0]
    assert get_origin(agg_tuple) is tuple, "Aggregate.aggs elements must be tuples"
    assert get_args(agg_tuple) == (AggCall, str), "Aggregate.aggs elements must be tuple[AggCall, str]"

    # 5. Sort.keys: list[tuple[Expr, bool]]
    sort_hints = get_type_hints(Sort)
    sort_keys = sort_hints["keys"]
    assert get_origin(sort_keys) is list, "Sort.keys must be strictly list"
    sort_tuple = get_args(sort_keys)[0]
    assert get_origin(sort_tuple) is tuple, "Sort.keys elements must be tuples"
    assert get_args(sort_tuple) == (Expr, bool), "Sort.keys elements must be tuple[Expr, bool]"


def test_expr_subclasses_exist_and_frozen() -> None:
    """Verifies Expr base class and its subclasses (ColumnRef, Literal, BinaryOp, UnaryOp, AggCall)."""
    assert inspect.isclass(Expr), "Expr must be a class"

    for cls in EXPR_CLASSES:
        assert issubclass(cls, Expr), f"{cls.__name__} must inherit from Expr"
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        params = getattr(cls, "__dataclass_params__", None)
        assert params is not None and params.frozen, f"{cls.__name__} must be a frozen dataclass"


def test_dtype_enum_members() -> None:
    """Verifies DType enum and its exact member set: INT, FLOAT, STRING, BOOL, DATE."""
    assert issubclass(DType, Enum), "DType must be an Enum"
    expected_members = {"INT", "FLOAT", "STRING", "BOOL", "DATE"}
    actual_members = set(DType.__members__.keys())
    assert actual_members == expected_members, f"DType members must be {expected_members}, got {actual_members}"


def test_catalog_interface_and_signatures() -> None:
    """Verifies Catalog interface methods and signature contracts."""
    assert inspect.isclass(Catalog), "Catalog must be a class"

    # catalog.schema(table: str) -> list[tuple[str, DType]]
    assert hasattr(Catalog, "schema"), "Catalog must provide schema method"
    schema_hints = get_type_hints(Catalog.schema)
    assert schema_hints.get("table") is str
    schema_ret = schema_hints.get("return")
    assert get_origin(schema_ret) is list
    tuple_t = get_args(schema_ret)[0]
    assert get_origin(tuple_t) is tuple
    assert get_args(tuple_t) == (str, DType)

    # catalog.row_count(table: str) -> int
    assert hasattr(Catalog, "row_count"), "Catalog must provide row_count method"
    rc_hints = get_type_hints(Catalog.row_count)
    assert rc_hints.get("table") is str
    assert rc_hints.get("return") is int

    # catalog.stats(table: str, column: str) -> ColumnStats
    assert hasattr(Catalog, "stats"), "Catalog must provide stats method"
    stats_hints = get_type_hints(Catalog.stats)
    assert stats_hints.get("table") is str
    assert stats_hints.get("column") is str
    assert stats_hints.get("return") is ColumnStats

    # ColumnStats dataclass fields: ndv, min, max, null_count
    assert dataclasses.is_dataclass(ColumnStats)
    cs_fields = {f.name: f for f in dataclasses.fields(ColumnStats)}
    assert set(cs_fields.keys()) == {"ndv", "min", "max", "null_count"}
    cs_hints = get_type_hints(ColumnStats)
    assert cs_hints["ndv"] is int
    assert cs_hints["null_count"] is int


def test_frontend_parse_and_bind_signature() -> None:
    """Verifies parse_and_bind signature and public export in frontend."""
    assert hasattr(frontend, "parse_and_bind"), "frontend package must expose parse_and_bind"
    fn = frontend.parse_and_bind
    assert callable(fn), "parse_and_bind must be callable"

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert params == ["sql", "catalog"], f"parse_and_bind parameters must be ['sql', 'catalog'], got {params}"

    hints = get_type_hints(fn)
    assert hints.get("sql") is str, "parse_and_bind sql argument must be str"
    assert hints.get("catalog") is Catalog, "parse_and_bind catalog argument must be Catalog"
    assert issubclass(hints.get("return"), PlanNode), "parse_and_bind must return PlanNode"


def test_runtime_behavioral_invariants() -> None:
    """Validates dynamic immutability, children, schema, and replace_children behaviors."""
    schema = [("x", DType.INT), ("y", DType.FLOAT)]
    scan = Scan(table="t", columns=["x"], pushed_predicate=None, table_schema=schema)

    # children property
    assert scan.children == ()

    # schema derivation
    assert scan.schema() == [("x", DType.INT)]

    # immutability: mutating field raises FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        scan.table = "other"  # type: ignore[misc]

    # replace_children produces a new instance
    new_scan = scan.replace_children(())
    assert new_scan is not scan
    assert new_scan.table == scan.table
    assert new_scan.columns == scan.columns
    assert new_scan.table_schema == scan.table_schema

    # Invalid replace_children raises ValueError
    filter_node = Filter(child=scan, predicate=Literal(value=True, dtype=DType.BOOL))
    assert filter_node.children == (scan,)
    assert filter_node.schema() == [("x", DType.INT)]

    with pytest.raises(ValueError):
        filter_node.replace_children(())
