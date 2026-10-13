"""Phase C7: demo.py and the generated sources committed in bench/samples/."""
import ast
import io
from pathlib import Path

import pytest

import demo
from bench import runner
from codegen.runner import compile_module
from runtime import compare_tables, interpret

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "bench" / "samples"
SECTIONS = ["1. SQL", "2. Bound plan", "3. Optimized plan", "4. Optimizer, pass by pass",
            "5. Generated code", "6. Result", "7. Timing"]


def run_main(argv, capsys):
    code = demo.main(argv)
    return code, capsys.readouterr().out


def test_demo_prints_every_stage_in_order(capsys):
    code, out = run_main(["--query", "q14", "--runs", "1", "--color", "never"], capsys)
    assert code == 0
    positions = [out.index(title) for title in SECTIONS]
    assert positions == sorted(positions)
    assert "def run(tables):" in out
    assert "identical to the reference interpreter" in out
    assert "naive: interpreted" in out and "compiled + optimizer" in out


def test_demo_takes_a_sql_string_and_explains_alias_resolution(capsys):
    sql = ("SELECT c.nation, COUNT(*) AS n FROM customer c JOIN orders o ON c.id = o.cust_id "
           "WHERE o.total_price > 100.0 GROUP BY c.nation ORDER BY n DESC LIMIT 3")
    code, out = run_main([sql, "--runs", "1", "--color", "never"], capsys)
    assert code == 0
    assert "alias qualifiers resolved to table names" in out
    assert any(f"c {arrow} customer, o {arrow} orders" in out for arrow in ("->", "→"))
    assert "3 rows" in out


def test_alias_note_only_when_the_sql_has_table_aliases(capsys):
    # a column named `s` is not a table alias
    sql = "SELECT nation, COUNT(*) AS s FROM customer GROUP BY nation ORDER BY s DESC LIMIT 2"
    code, out = run_main([sql, "--runs", "1", "--color", "never"], capsys)
    assert code == 0 and "alias qualifiers" not in out
    assert demo.sql_aliases("SELECT x.id FROM orders x JOIN customer ON x.cust_id = customer.id") \
        == {"x": "orders"}


def test_ascii_mode_without_colour_is_plain_ascii(capsys):
    code, out = run_main(["--query", "q01", "--runs", "1", "--ascii", "--color", "never"], capsys)
    assert code == 0
    out.encode("ascii")            # raises if anything non-ASCII slipped through
    assert "\033[" not in out


def test_colour_codes_only_when_asked(capsys):
    _, out = run_main(["--query", "q01", "--runs", "1", "--color", "always"], capsys)
    assert "\033[" in out


def test_list_and_usage_errors(capsys):
    code, out = run_main(["--list"], capsys)
    assert code == 0 and len(out.strip().splitlines()) == 20
    with pytest.raises(SystemExit):
        demo.main([])
    with pytest.raises(SystemExit):
        demo.main(["--query", "q99"])


def test_wrong_answer_makes_the_demo_fail_loudly(monkeypatch, capsys):
    from runtime._compat import Limit
    monkeypatch.setattr(runner, "find_optimizer",
                        lambda: (lambda plan, catalog: (Limit(child=plan, n=0), []), "broken"))
    code, out = run_main(["--query", "q01", "--runs", "1", "--color", "never"], capsys)
    assert code == 1 and "DIFFERENT from the reference interpreter" in out


# ------------------------------------------------------------------ bench/samples

SAMPLES = sorted(SAMPLE_DIR.glob("q*.py"))


def test_five_samples_are_committed():
    assert [p.name.split("_")[0] for p in SAMPLES] == sorted(demo.SAMPLES)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_committed_sample_runs_and_gives_the_right_answer(path):
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    catalog, tables = runner.load_data("tiny")
    name = path.name.split("_")[0]
    sql = runner.load_queries([name])[name]
    assert sql.splitlines()[0] in source                       # the header quotes the SQL
    result = compile_module(source)["run"](tables)
    ok, why = compare_tables(interpret(runner.bind(sql, catalog), tables), result,
                             ordered=runner.is_ordered(sql))
    assert ok, why


def test_write_samples(tmp_path):
    written = demo.write_samples(tmp_path)
    assert sorted(p.name for p in written) == sorted(f"{q}_{s}.py" for q, s in demo.SAMPLES.items())
