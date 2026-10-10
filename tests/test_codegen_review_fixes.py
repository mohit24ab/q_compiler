"""Fixes for Person B's review of the code generator and runtime (Oct 2026).

1. A SQL string literal holding a newline broke out of a comment in the generated code
   and ran as Python.
2. Constant GROUP BY / join keys (`GROUP BY 'all'`, `GROUP BY nation, 1+1`) crashed
   generated code.
3. compare_tables rounded floats to 6 decimals.
4. A relation with rows but no columns lost its row count.
5. DATE vs STRING comparisons disagreed between the engines.
6. The stand-in IR is gone.
7. (Since A's binder accepts `||` on any type, main @ a2ed35c) generated code concatenated
   only STRINGs; `id || name` crashed.

The differential suites run every plan in codegen_fixtures.REVIEW_PLANS through both
engines; the tests here pin the answers by hand, so a bug shared by both engines shows.
"""
import ast
import builtins
import datetime
from pathlib import Path

import pytest

from codegen import compile_and_run, generate
from codegen.emitter import Emitter
from runtime import Table, compare_tables, interpret
from runtime._compat import AggCall, Aggregate, DType, Filter, Join, Project, Sort, format_plan

from codegen_fixtures import CATALOG, REVIEW_PLANS, TABLES, col, lit, op, scan

MARKER = "CODEGEN_INJECTED"


def every_engine(plan, catalog=CATALOG, tables=TABLES):
    """The interpreter's answer, then generated code's, fused and not."""
    return [interpret(plan, tables)] + [
        compile_and_run(generate(plan, catalog, mode="compiled", fuse=f), tables)
        for f in (True, False)]


def rows(name):
    """Every engine's rows for a REVIEW_PLANS plan; asserts they agree."""
    results = [r.to_rows() for r in every_engine(REVIEW_PLANS[name][0])]
    assert all(r == results[0] for r in results), results
    return results[0]


# ------------------------------------------------------------------ 1. code injection

PAYLOADS = [
    f"\n__import__('builtins').__dict__.setdefault({MARKER!r}, 1)\n#",
    f"\r__import__('builtins').__dict__.setdefault({MARKER!r}, 2)\r",
    f"\r\n__import__('builtins').__dict__.setdefault({MARKER!r}, 3)",
    f'"""\n__import__("builtins").__dict__.setdefault({MARKER!r}, 4)\n"""',
    f"\\\n__import__('builtins').__dict__.setdefault({MARKER!r}, 5)",
    f" \x0c\x00__import__('builtins').__dict__.setdefault({MARKER!r}, 6)",
]


def _plans_quoting(text):
    """Plans that put `text` into every place plan text reaches the generated source:
    aggregate labels (grouped and global), group keys, fused and unfused predicates,
    a LIKE pattern, sort keys, aliases, the docstring."""
    s = lit(text, DType.STRING)
    region = col("region", "sales")
    return {
        "global_count": Aggregate(child=scan("sales"), group_keys=[],
                                  aggs=[(AggCall("count", s), "n"), (AggCall("min", s), "m")]),
        "grouped_min": Aggregate(child=scan("sales"), group_keys=[region, s],
                                 aggs=[(AggCall("max", s), text), (AggCall("count", None), "n")]),
        "fused_filter": Project(child=Filter(child=scan("sales"), predicate=op("<>", region, s)),
                                exprs=[(op("||", region, s), text)]),
        "like_pattern": Filter(child=scan("sales"), predicate=op("NOT LIKE", region, s)),
        "pushed_and_sorted": Sort(child=scan("sales", pred=op("<>", region, s)),
                                  keys=[(op("||", region, s), True)]),
        "join_residual": Join(left=scan("orders"), right=scan("customer"),
                              condition=op("AND", op("=", col("cust_id", "orders"), col("id", "customer")),
                                           op("<>", col("name", "customer"), s)), kind="left"),
    }


@pytest.mark.parametrize("payload", range(len(PAYLOADS)))
@pytest.mark.parametrize("mode", ["compiled", "passthrough"])
def test_string_literals_in_plan_text_never_become_code(payload, mode):
    text = PAYLOADS[payload]
    for name, plan in _plans_quoting(text).items():
        for fuse in (True, False):
            src = generate(plan, CATALOG, mode=mode, fuse=fuse)
            tree = ast.parse(src)   # it compiles at all: nothing broke out half-way
            calls = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            assert "__import__" not in calls, f"{name}: payload is code\n{src}"
            doc = ast.get_docstring(tree, clean=False)   # the plan, character for character
            assert all(line in doc for line in format_plan(plan).splitlines())
            assert all(ch not in src for ch in "\r\x00\x0c ")
            out = compile_and_run(src, TABLES)
            assert MARKER not in builtins.__dict__, f"{name}: the payload ran\n{src}"
            ok, why = compare_tables(interpret(plan, TABLES), out)
            assert ok, f"{name}: {why}"


def test_comments_escape_what_would_end_the_line():
    em = Emitter()
    with em.block("def run(tables):"):
        em.comment("sum('a\nb')\r\x00")
        em.line("x = 1", note="count('\n')")
        em.line("return x")
    body = em.body_text().splitlines()
    assert body[1] == r"    # sum('a\nb')\r\x00"
    assert body[2] == r"    x = 1  # count('\n')"


def test_emitter_refuses_a_line_that_spans_lines():
    for bad in ("x = 1\ny = 2", "x = 1\r", "x = '\x00'"):
        with pytest.raises(ValueError, match="one physical line"):
            Emitter().line(bad)


# ------------------------------------------------------------------ 2. constant keys

def test_group_by_only_constants_is_one_group_or_none():
    # sales: 6 rows, qty 1..6
    assert rows("group_by_constant_only") == [("all", 6, 21)]
    assert rows("group_by_constant_over_no_rows") == []


def test_group_by_column_and_constants_groups_by_the_column():
    got = sorted(rows("group_by_column_and_constants"), key=lambda r: (r[0] is None, r[0] or ""))
    assert got == [("AP", 2, None, 1), ("EU", 2, None, 2), ("US", 2, None, 2), (None, 2, None, 1)]


def test_constant_key_whose_nullness_is_computed_at_run_time():
    got = {r[1]: r for r in rows("group_by_constant_null_only_known_at_run_time")}
    assert got["US"] == (None, "US", 2) and got[None] == (None, None, 1)
    assert rows("group_by_key_null_on_every_row") == [(None, 6)]


def test_constant_join_keys():
    assert rows("join_on_constant_null_key_inner") == []                 # NULL never matches
    left = rows("join_on_constant_null_key_left")
    assert [r[0] for r in left] == [100, 101, 102, 103]                  # every left row, padded
    assert all(r[3:] == (None, None, None) for r in left)
    # 3 cust_id groups, each joined with the 3 payments whose amt > 4 (5, 7, 11)
    pairs = rows("join_on_constant_true_key")
    assert len(pairs) == 9 and {r[3] for r in pairs} == {5, 7, 11}


SQL_CASES = [
    "SELECT 'all' AS k, COUNT(*) AS n FROM customer GROUP BY 'all'",
    "SELECT nation, COUNT(*) AS n FROM customer GROUP BY nation, 1+1",
    "SELECT COUNT(*) AS n FROM customer c JOIN orders o ON c.id / 0 = o.cust_id",
    "SELECT COUNT(*) AS n FROM customer c LEFT JOIN orders o ON c.id / 0 = o.cust_id",
    "SELECT COUNT('x\nprint(1)\n') AS c, MIN('a\n#') AS m FROM customer",
    "SELECT nation, MAX('\"\"\"\n') AS m, COUNT(*) AS n FROM customer GROUP BY nation, 'a\nb'",
    "SELECT id || '-' || name AS s, acctbal || '' AS a, (id > 3) || nation AS b FROM customer",
]


@pytest.mark.parametrize("sql", SQL_CASES)
def test_review_queries_from_sql_unoptimized_and_optimized(sql):
    from bench import runner
    from frontend.binder import parse_and_bind
    from optimizer import optimize
    catalog, tables = runner.load_data("tiny", 42)
    plan = parse_and_bind(sql, catalog)
    for p in (plan, optimize(plan, catalog)[0]):
        expected, *compiled = every_engine(p, catalog, tables)
        assert expected.num_rows > 0
        for got in compiled:
            ok, why = compare_tables(expected, got)
            assert ok, f"{sql}: {why}"


# ------------------------------------------------------------------ 3. float comparison

def _floats(*values):
    return Table.from_pydict({"x": list(values)}, [("x", DType.FLOAT)])


def _pairs(*rows_):
    return Table.from_rows([("k", DType.STRING, None), ("x", DType.FLOAT, None)], rows_)


def test_large_sums_that_differ_by_summation_noise_are_equal():
    # rounded to 6 decimals these were ...123456 vs ...123458: reported as different
    big = 1234567890.123456
    assert compare_tables(_floats(big), _floats(big + 2e-6), ordered=True)[0]
    assert compare_tables(_floats(0.4999995), _floats(0.49999950000000004), ordered=True)[0]


def test_small_values_that_really_differ_are_not_equal():
    # rounded to 6 decimals both were 0.0: reported as equal
    assert not compare_tables(_floats(1e-7), _floats(4e-7), ordered=True)[0]
    assert not compare_tables(_floats(1.0), _floats(1.00001), ordered=True)[0]


def test_nan_null_and_order_with_noisy_floats():
    assert compare_tables(_floats(float("nan"), None), _floats(None, float("nan")))[0]
    assert not compare_tables(_floats(float("nan")), _floats(0.0))[0]
    # near-equal floats in different groups don't pair up the wrong rows
    a = _pairs(("a", 5.0000000001), ("b", 5.0))
    b = _pairs(("b", 5.0 + 1e-15), ("a", 5.0000000001 - 1e-15))
    assert compare_tables(a, b)[0]
    assert not compare_tables(a, _pairs(("a", 5.0), ("b", 6.0)))[0]


# ------------------------------------------------------------------ 4. no columns, some rows

def test_table_with_no_columns_keeps_its_row_count():
    t = Table([], 3)
    assert t.num_rows == 3 and t.to_rows() == [(), (), ()]
    assert t.take([0, 2]).num_rows == 2 and t.select([]).num_rows == 3
    assert Table.concat([t, Table([], 2)]).num_rows == 5
    assert Table.from_rows([], [(), ()]).num_rows == 2
    assert Table([]).num_rows == 0                       # the old default is unchanged
    with pytest.raises(ValueError, match="differing lengths"):
        Table(_floats(1.0, 2.0).columns, 3)


def test_counting_rows_through_operators_with_no_columns():
    # sales: 6 rows, qty > 2 keeps 4; orders: 4 rows
    assert rows("count_over_scan_with_no_columns") == [(6,)]
    assert rows("count_over_pushed_scan_with_no_columns") == [(4,)]
    assert rows("count_over_filter_over_no_columns") == [(6,)]
    assert rows("count_over_project_with_no_exprs") == [(6,)]
    assert rows("count_over_join_of_no_columns") == [(24,)]
    assert rows("count_over_limit_of_no_columns") == [(4,)]
    assert rows("group_by_over_no_columns") == [(7, 6)]
    assert rows("scan_with_no_columns") == [()] * 6


# ------------------------------------------------------------------ 5. DATE vs STRING

def test_date_compared_with_string_parses_the_string_as_an_iso_date():
    # day: 01-05, 03-01, NULL, 02-02, 01-05     txt: 01-05, 02-01, 02-02, NULL, '20240105'
    assert rows("date_column_equals_string_column") == [(1,), (5,)]
    assert rows("string_column_before_date_column") == [(2,)]
    assert rows("date_column_equals_iso_basic_string") == [(1,), (5,)]
    assert rows("iso_basic_date_literal_equals_date_column") == [(1,), (5,)]
    assert rows("string_column_after_date_literal") == [(2,), (3,)]


def test_strings_that_are_not_dates_fail_alike_and_only_when_compared():
    t = Table.from_pydict(
        {"day": ["2024-01-05", None], "txt": ["2024-01-05", "not a date"]},
        [("day", DType.DATE), ("txt", DType.STRING)])
    tables = {"ev": t}

    class Catalog:
        def schema(self, table):
            return list(tables[table].schema)

    # 'not a date' sits next to a NULL day: neither engine ever parses it
    plan = Filter(child=scan("ev"), predicate=op("=", col("day"), col("txt")))
    for result in every_engine(plan, Catalog(), tables):
        assert result.to_rows() == [(datetime.date(2024, 1, 5), "2024-01-05")]
    # compared with a literal it is parsed, and both engines reject it
    bad = Filter(child=scan("ev"), predicate=op(">", col("txt"), lit("2024-01-01", DType.DATE)))
    with pytest.raises(Exception, match="Invalid isoformat"):
        interpret(bad, tables)
    with pytest.raises(Exception, match="Invalid isoformat"):
        compile_and_run(generate(bad, Catalog(), mode="compiled"), tables)


# ------------------------------------------------------------------ 7. || on any type

def test_concat_writes_each_value_as_the_interpreter_does():
    # sales row 1: id 1, region US, amount 10.0, qty 1, day 2024-01-01
    assert rows("concat_every_type")[0] == (
        "1US", "10.0!", "False2024-01-01", "72.5True", "2024-01-051")
    assert rows("concat_every_type")[3][1] is None      # amount is NULL: so is the result


# ------------------------------------------------------------------ 6. the real IR only

def test_runtime_uses_person_a_ir_and_the_stand_in_is_gone():
    import ir.nodes
    import runtime._compat as compat
    assert compat.Scan is ir.nodes.Scan
    assert not (Path(compat.__file__).parent / "_ir_standin.py").exists()
