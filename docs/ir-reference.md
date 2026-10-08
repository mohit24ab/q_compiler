# Relational Algebra Intermediate Representation (IR) Specification

> **Status:** Contract Freeze (Phase A6)  
> **Audience:** Person B (Optimizer Engineer) & Person C (Codegen/Runtime Engineer)  
> **Owner:** Person A (Frontend & IR Architect)  

---

## 1. Overview & Architectural Invariants

The `q_compiler` Intermediate Representation (IR) represents relational query execution plans as a tree of immutable relational algebra operator nodes. The IR bridges the SQL frontend (`frontend/binder.py`) with the rule-based and cost-based optimizer framework (`optimizer/`) and backend code generators/interpreters (`codegen/`, `runtime/`).

### 1.1 Immutability Invariant
All IR plan nodes (`ir.nodes.PlanNode`) and expression nodes (`ir.expr.Expr`) are defined as **frozen Python dataclasses** (`@dataclass(frozen=True)`).
- **No in-place mutation:** Modifying attributes on any node instance raises `dataclasses.FrozenInstanceError`.
- **Pure Functional Transformations:** Optimization passes and IR rewrites must return new plan node instances rather than mutating existing trees in place.
- **Structural Hashing & Equality:** Because all nodes are frozen, nodes implement structural equality (`__eq__`) and hashing (`__hash__`), making them safe for memoization, graph caches, and set memberships.

### 1.2 Tree Rewriting Protocol
Every plan node exposes two standard transformation primitives:
1. `children: tuple[PlanNode, ...]`: An immutable property exposing immediate child nodes.
2. `replace_children(new_children: Sequence[PlanNode]) -> PlanNode`: A constructor method that validates child arity and returns a new node with updated children while preserving node-specific metadata.

Tree traversals should be implemented using the post-order visitor helpers in `ir.visitor`:
```python
from ir.visitor import transform_post_order, transform_expr_post_order

# Bottom-up plan transformation:
new_plan = transform_post_order(root_plan, rewrite_rule_fn)
```

---

## 2. Type System (`ir.dtype.DType`)

The compiler implements a strict, static relational type system defined by `ir.dtype.DType`:

| `DType` Member | Description | Physical PyArrow Mapping | Python Primitive |
| :--- | :--- | :--- | :--- |
| `DType.INT` | 64-bit signed integer | `pa.int64()`, `pa.int32()` | `int` |
| `DType.FLOAT` | 64-bit IEEE double float | `pa.float64()` | `float` |
| `DType.STRING` | UTF-8 encoded text string | `pa.string()`, `pa.large_string()` | `str` |
| `DType.BOOL` | Boolean truth value | `pa.bool_()` | `bool` |
| `DType.DATE` | Calendar date (days since epoch) | `pa.date32()` | `datetime.date`, `str` (ISO-8601) |

### Type Coercion & Promotion Rules:
1. **Arithmetic (`+`, `-`, `*`, `%`):**
   - `INT` op `INT` $\rightarrow$ `INT`
   - `INT` op `FLOAT` or `FLOAT` op `INT` $\rightarrow$ `FLOAT`
   - `FLOAT` op `FLOAT` $\rightarrow$ `FLOAT`
2. **Division (`/`):**
   - Numeric op Numeric $\rightarrow$ `FLOAT` (always produces floating-point result)
3. **Comparisons (`=`, `!=`, `<`, `<=`, `>`, `>=`):**
   - Numeric $\leftrightarrow$ Numeric $\rightarrow$ `BOOL`
   - `DATE` $\leftrightarrow$ `DATE` $\rightarrow$ `BOOL`
   - `DATE` $\leftrightarrow$ `STRING` (ISO date literal) $\rightarrow$ `BOOL`
   - `STRING` $\leftrightarrow$ `STRING` $\rightarrow$ `BOOL`
4. **Logical (`AND`, `OR`, `NOT`):**
   - Operands must evaluate to `BOOL` $\rightarrow$ `BOOL`

---

## 3. Plan Node Specifications (`ir.nodes`)

All plan nodes inherit from the abstract base class `PlanNode`:
```python
class PlanNode(ABC):
    @property
    @abstractmethod
    def children(self) -> tuple[PlanNode, ...]: ...

    @abstractmethod
    def schema(self) -> list[tuple[str, DType]]: ...

    @abstractmethod
    def replace_children(self, new_children: Sequence[PlanNode]) -> PlanNode: ...
```

### 3.1 `Scan`
Leaf node representing a table scan over physical catalog storage.

```python
@dataclass(frozen=True)
class Scan(PlanNode):
    table: str
    columns: list[str] | None
    pushed_predicate: Expr | None
    table_schema: list[tuple[str, DType]] | None = None
```

- **Fields:**
  - `table: str`: Name of the source table in the `Catalog`.
  - `columns: list[str] | None`: Column projection list. In canonical unoptimized plans, this is `None` (representing a full scan of all columns). Populated by the column pruning optimization pass.
  - `pushed_predicate: Expr | None`: Boolean filter pushed down directly to the physical scan. In unoptimized plans, this is `None`. Populated by the predicate pushdown pass.
  - `table_schema: list[tuple[str, DType]] | None`: Full catalog schema registered at bind time. Default is `None` (must be populated for schema queries).
- **Children:** `()` (0 children).
- **Schema Derivation:**
  - If `columns is None`: Returns `table_schema`.
  - If `columns` is specified: Returns `[(col, dtype)]` preserving the exact order requested in `columns`. Raises `ValueError` if a column does not exist in `table_schema`.
- **replace_children:** Expects empty sequence `()`. Raises `ValueError` if `len(new_children) != 0`.
- **Printer Format:**
  - Unoptimized: `Scan[customer]`
  - Pruned / Pushed: `Scan[customer, columns=[id, acctbal], pushed=customer.acctbal > 0.0]`

---

### 3.2 `Filter`
Applies a boolean predicate to filter rows emitted by its child operator.

```python
@dataclass(frozen=True)
class Filter(PlanNode):
    child: PlanNode
    predicate: Expr
```

- **Fields:**
  - `child: PlanNode`: Upstream input operator producing input rows.
  - `predicate: Expr`: Filter expression. Must evaluate to `DType.BOOL`.
- **Children:** `(child,)` (1 child).
- **Schema Derivation:** Pass-through; returns `child.schema()`.
- **replace_children:** Expects sequence of length 1 `(new_child,)`.
- **Printer Format:** `Filter[customer.mktsegment = 'BUILDING']`

---

### 3.3 `Project`
Computes scalar projections, arithmetic expressions, or column aliases.

```python
@dataclass(frozen=True)
class Project(PlanNode):
    child: PlanNode
    exprs: list[tuple[Expr, str]]
```

- **Fields:**
  - `child: PlanNode`: Input operator.
  - `exprs: list[tuple[Expr, str]]`: List of `(expression, output_alias)` pairs.
- **Children:** `(child,)` (1 child).
- **Schema Derivation:** Evaluates each expression against `child.schema()`, returning `[(alias, inferred_dtype), ...]`.
- **replace_children:** Expects sequence of length 1 `(new_child,)`.
- **Printer Format:** `Project[customer.id AS id, customer.acctbal * 1.1 AS adjusted_bal]`

---

### 3.4 `Join`
Joins two input relations based on a binary join predicate.

```python
@dataclass(frozen=True)
class Join(PlanNode):
    left: PlanNode
    right: PlanNode
    condition: Expr
    kind: Literal["inner", "left"]
```

- **Fields:**
  - `left: PlanNode`: Outer/left input relation.
  - `right: PlanNode`: Inner/right input relation.
  - `condition: Expr`: Join predicate (typically equality `left.fk = right.pk`).
  - `kind: Literal["inner", "left"]`: Join type.
- **Children:** `(left, right)` (2 children).
- **Schema Derivation:** Concatenation: `left.schema() + right.schema()`.
- **replace_children:** Expects sequence of length 2 `(new_left, new_right)`.
- **Printer Format:** `Join[kind=inner, cond=o.cust_id = c.id]`

---

### 3.5 `Aggregate`
Performs grouping and computes aggregate functions.

```python
@dataclass(frozen=True)
class Aggregate(PlanNode):
    child: PlanNode
    group_keys: list[Expr]
    aggs: list[tuple[AggCall, str]]
```

- **Fields:**
  - `child: PlanNode`: Input operator.
  - `group_keys: list[Expr]`: Grouping expressions (typically `ColumnRef` nodes). Empty for global scalar aggregations.
  - `aggs: list[tuple[AggCall, str]]`: List of aggregate function invocations with assigned output aliases `(AggCall, alias)`.
- **Children:** `(child,)` (1 child).
- **Schema Derivation:**
  - First, grouping keys with their names and inferred types: `[(key_name, key_dtype), ...]`.
  - Next, aggregate calls with their aliases and return types: `[(alias, agg_dtype), ...]`.
- **replace_children:** Expects sequence of length 1 `(new_child,)`.
- **Printer Format:** `Aggregate[group=customer.nation, aggs=count(*) AS cust_count, avg(customer.acctbal) AS avg_bal]`

---

### 3.6 `Sort`
Orders the input dataset by one or more ordering expressions.

```python
@dataclass(frozen=True)
class Sort(PlanNode):
    child: PlanNode
    keys: list[tuple[Expr, bool]]
```

- **Fields:**
  - `child: PlanNode`: Input operator.
  - `keys: list[tuple[Expr, bool]]`: Ordered list of `(sort_expression, is_descending)` pairs (`True` for `DESC`, `False` for `ASC`).
- **Children:** `(child,)` (1 child).
- **Schema Derivation:** Pass-through; returns `child.schema()`.
- **replace_children:** Expects sequence of length 1 `(new_child,)`.
- **Printer Format:** `Sort[keys=customer.acctbal DESC, customer.id ASC]`

---

### 3.7 `Limit`
Truncates row output to at most `n` records.

```python
@dataclass(frozen=True)
class Limit(PlanNode):
    child: PlanNode
    n: int
```

- **Fields:**
  - `child: PlanNode`: Input operator.
  - `n: int`: Maximum number of rows to emit ($n \ge 0$).
- **Children:** `(child,)` (1 child).
- **Schema Derivation:** Pass-through; returns `child.schema()`.
- **replace_children:** Expects sequence of length 1 `(new_child,)`.
- **Printer Format:** `Limit[n=10]`

---

## 4. Expression Node Specifications (`ir.expr`)

All expression nodes inherit from `ir.expr.Expr` and are frozen dataclasses.

### 4.1 `ColumnRef`
References a relation attribute by name and optional qualifying table/alias.
```python
@dataclass(frozen=True)
class ColumnRef(Expr):
    table: str | None
    name: str
```
- **Type Inference:** Looked up from input relation schema matching `name` or `f"{table}.{name}"`.

### 4.2 `Literal`
A constant scalar value.
```python
@dataclass(frozen=True)
class Literal(Expr):
    value: Any
    dtype: DType
```
- **Type Inference:** Returns explicitly defined `dtype`.

### 4.3 `BinaryOp`
Binary operator evaluating left and right sub-expressions.
```python
@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr
```
- **Supported Operators:**
  - Arithmetic: `+`, `-`, `*`, `/`, `%`
  - Comparisons: `=`, `!=`, `<`, `<=`, `>`, `>=`
  - Logical: `AND`, `OR`
- **Precedence Hierarchy:**
  1. `OR` (precedence 1)
  2. `AND` (precedence 2)
  3. Comparison: `=`, `!=`, `<`, `<=`, `>`, `>=` (precedence 3)
  4. Addition/Subtraction: `+`, `-` (precedence 4)
  5. Multiplication/Division/Modulo: `*`, `/`, `%` (precedence 5)

### 4.4 `UnaryOp`
Unary prefix or postfix operator.
```python
@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    operand: Expr
```
- **Supported Operators:**
  - Logical negation: `NOT` (requires boolean operand; returns `DType.BOOL`).
  - Null testing: `IS NULL`, `IS NOT NULL` (accepts any operand; returns `DType.BOOL`).
  - Arithmetic negation: `-` (requires numeric operand; returns operand type).

### 4.5 `AggCall`
Aggregate function invocation over an argument expression.
```python
@dataclass(frozen=True)
class AggCall(Expr):
    func: Literal["sum", "count", "avg", "min", "max"]
    arg: Expr | None
```
- `count`: `arg` is `None` for `COUNT(*)` or an expression for `COUNT(col)`. Always returns `DType.INT`.
- `sum`: Requires numeric `arg`. Returns `arg.dtype`.
- `avg`: Requires numeric `arg`. Always returns `DType.FLOAT`.
- `min`: Accepts numeric, string, or date `arg`. Returns `arg.dtype`.
- `max`: Accepts numeric, string, or date `arg`. Returns `arg.dtype`.

---

## 5. Catalog and Statistics Interface (`catalog`)

The `Catalog` manages registered physical tables and metadata:

```python
class Catalog:
    def schema(self, table: str) -> list[tuple[str, DType]]:
        """Returns table schema as a list of (column_name, DType) tuples."""

    def row_count(self, table: str) -> int:
        """Returns the total row count for a table."""

    def stats(self, table: str, column: str) -> ColumnStats:
        """Returns column statistics (NDV, min, max, null_count) for a table column."""
```

### Statistics Contract (`ColumnStats`):
```python
@dataclass(frozen=True)
class ColumnStats:
    ndv: int         # Number of distinct values in the column
    min: Any         # Minimum value (ISO string for dates, None if all null)
    max: Any         # Maximum value (ISO string for dates, None if all null)
    null_count: int  # Number of null values in the column
```

---

## 6. Canonical Unoptimized Plan Hierarchy

The SQL binder (`frontend.binder.parse_and_bind(sql, catalog)`) transforms SQL queries into a strictly canonical, unoptimized plan tree structure.

### Operator Layering Hierarchy (Top to Bottom):
```text
Limit                     [Topmost; present if LIMIT clause was specified]
  Sort                    [Present if ORDER BY clause was specified]
    Project               [SELECT projections and expressions]
      Filter (HAVING)     [Present if HAVING clause was specified]
        Aggregate         [Present if GROUP BY or aggregate functions exist]
          Filter (WHERE)  [Present if WHERE clause was specified]
            Join          [Left-associative tree; present if JOINs exist]
              Join
                Scan
                Scan
              Scan
            Scan          [Leaf nodes]
```

### Invariants of Unoptimized Plans:
1. **Unpruned Scans:** Every `Scan` node has `columns=None`.
2. **Unpushed Predicates:** Every `Scan` node has `pushed_predicate=None`.
3. **Populated Table Schemas:** Every `Scan` node has `table_schema` set from the `Catalog`.
4. **Normalized Join Structure:** Multi-table joins are left-associative binary joins (`Join(left=Join(left=..., right=Scan), right=Scan)`).
5. **Separation of Filters:** Filter predicates are cleanly partitioned between WHERE filters (below aggregates) and HAVING filters (above aggregates).

---

## 7. Developer Handoff Checklist

### For Person B (Optimizer):
- **Rule-Based Passes:** Use `transform_post_order` to rewrite trees. Replace child nodes using `node.replace_children(...)`.
- **Column Pruning Pass:** Walk top-down, accumulate required columns, and update `Scan.columns = [...]`.
- **Predicate Pushdown Pass:** Push down filter conjunctions into `Scan.pushed_predicate` or join conditions.
- **Cost-Based Join Reordering:** Query `catalog.row_count(table)` and `catalog.stats(table, col)` to estimate intermediate cardinalities.

### For Person C (Codegen & Runtime):
- **Interpreted Pipeline:** Implement recursive iterator generators (`open()`, `next()`, `close()`) conforming to each `PlanNode` subclass.
- **Execution Codegen:** Compile operators into tight vectorized kernels over PyArrow RecordBatches or NumPy arrays.
- **Schema Guarantee:** Rely on `plan.schema()` as the authoritative contract for tuple layouts and data types.
