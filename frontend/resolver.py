from __future__ import annotations

from typing import Sequence
from catalog.catalog import Catalog
from ir.dtype import DType
from ir.expr import ColumnRef


class SemanticError(ValueError):
    """Base exception for semantic and type analysis errors in query compilation."""

    pass


class Resolver:
    """Manages identifier scope, relation aliases, and column resolution."""

    def __init__(self, catalog: Catalog | None = None) -> None:
        self.catalog = catalog
        self.schemas: dict[str, list[tuple[str, DType]]] = {}
        self.table_to_alias: dict[str, str] = {}
        self.table_to_aliases: dict[str, list[str]] = {}
        self.alias_to_table: dict[str, str] = {}
        self.physical_tables: dict[str, str] = {}
        self.relation_order: list[str] = []

    def add_table(self, table_name: str, alias: str | None = None) -> list[tuple[str, DType]]:
        """Resolves a table from the catalog and registers it in the active scope."""
        if self.catalog is None:
            raise SemanticError("Catalog is required to resolve table schema.")
        try:
            schema = self.catalog.schema(table_name)
        except KeyError as exc:
            raise SemanticError(f"Table '{table_name}' not found in catalog.") from exc

        self.add_relation(table_name=table_name, schema=schema, alias=alias)
        return schema

    def add_relation(
        self,
        table_name: str,
        schema: Sequence[tuple[str, DType]],
        alias: str | None = None,
    ) -> None:
        """Registers a relation schema and optional alias in the active scope."""
        rel_key = alias if alias else table_name
        self.schemas[rel_key] = list(schema)
        self.relation_order.append(rel_key)
        self.physical_tables[rel_key] = table_name
        if alias:
            self.table_to_alias[table_name] = alias
            self.table_to_aliases.setdefault(table_name, []).append(alias)
            self.alias_to_table[alias] = table_name

    def get_canonical_table(self, target_rel: str) -> str:
        """Returns the canonical table name or alias to use in ColumnRef.

        For self-joins where the physical table is registered under multiple aliases,
        preserves the alias so scopes remain strictly isolated.
        """
        phys_table = self.physical_tables.get(target_rel, target_rel)
        occurrences = [r for r, phys in self.physical_tables.items() if phys == phys_table]
        if len(occurrences) > 1:
            return target_rel
        return self.alias_to_table.get(target_rel, target_rel)

    def resolve_column(self, name: str, table: str | None = None) -> ColumnRef:
        """Resolves a qualified or unqualified column reference against active schemas."""
        if table is not None:
            target_rel = table
            if target_rel not in self.schemas:
                aliases = self.table_to_aliases.get(table, [])
                if len(aliases) > 1:
                    raise SemanticError(
                        f"Ambiguous table reference '{table}' across multiple aliases: {aliases}."
                    )
                elif len(aliases) == 1 and aliases[0] in self.schemas:
                    target_rel = aliases[0]
                elif table in self.table_to_alias and self.table_to_alias[table] in self.schemas:
                    target_rel = self.table_to_alias[table]
                else:
                    raise SemanticError(f"Table '{table}' not found in active scope.")

            rel_schema = self.schemas[target_rel]
            col_names = [col_name for col_name, _ in rel_schema]
            if name not in col_names:
                raise SemanticError(f"Column '{name}' not found in table '{table}'.")
            canonical_table = self.get_canonical_table(target_rel)
            return ColumnRef(table=canonical_table, name=name)
        else:
            matches: list[str] = []
            for rel_name, rel_schema in self.schemas.items():
                col_names = [col_name for col_name, _ in rel_schema]
                if name in col_names:
                    matches.append(rel_name)

            if len(matches) == 0:
                raise SemanticError(f"Unknown column '{name}'.")
            if len(matches) > 1:
                raise SemanticError(
                    f"Ambiguous column reference '{name}' found across active tables: {matches}."
                )
            canonical_table = self.get_canonical_table(matches[0])
            return ColumnRef(table=canonical_table, name=name)

    def get_column_type(self, col: ColumnRef | str, table: str | None = None) -> DType:
        """Returns the DType for a qualified or unqualified column reference."""
        if isinstance(col, str):
            col_ref = self.resolve_column(name=col, table=table)
        else:
            col_ref = col

        if col_ref.table is not None:
            target_rel = col_ref.table
            if target_rel not in self.schemas:
                aliases = self.table_to_aliases.get(target_rel, [])
                if len(aliases) == 1 and aliases[0] in self.schemas:
                    target_rel = aliases[0]
                elif target_rel in self.table_to_alias and self.table_to_alias[target_rel] in self.schemas:
                    target_rel = self.table_to_alias[target_rel]
            if target_rel in self.schemas:
                for c_name, dt in self.schemas[target_rel]:
                    if c_name == col_ref.name:
                        return dt
            raise SemanticError(f"Column '{col_ref.name}' not found in table '{col_ref.table}'.")
        else:
            matches: list[DType] = []
            for rel_name, rel_schema in self.schemas.items():
                for c_name, dt in rel_schema:
                    if c_name == col_ref.name:
                        matches.append(dt)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise SemanticError(
                    f"Ambiguous column reference '{col_ref.name}' found across active tables."
                )
            raise SemanticError(f"Unknown column '{col_ref.name}'.")

    def get_schema(self, table_or_alias: str) -> list[tuple[str, DType]]:
        """Returns the schema for a table or alias."""
        target = table_or_alias
        if target not in self.schemas:
            aliases = self.table_to_aliases.get(target, [])
            if len(aliases) == 1 and aliases[0] in self.schemas:
                target = aliases[0]
            elif target in self.table_to_alias and self.table_to_alias[target] in self.schemas:
                target = self.table_to_alias[target]
        if target not in self.schemas:
            raise SemanticError(f"Table '{table_or_alias}' not found in active scope.")
        return list(self.schemas[target])
