"""
ConfigLoaderMixin: chunked loading of configurations and related entities into Neo4j.
Uses prebuilt rows from RowsBuilderMixin and reusable Cypher from cypher_templates.
"""
from __future__ import annotations

from typing import Any, Dict, List
import logging

from neo4j.exceptions import Neo4jError

from config import settings
from parsers.metadata_parser import Configuration

from .cypher_templates import (
    CYPHER_MERGE_PROJECT,
    CYPHER_UPSERT_CONFIGURATION,
    CYPHER_UPSERT_CATEGORIES,
    CYPHER_UPSERT_METADATA_OBJECT,
    CYPHER_UPSERT_SUBSYSTEM_EDGE,
    CYPHER_REPAIR_SUBSYSTEM_CATEGORY_EDGES,
    CYPHER_UPSERT_FORM,
    CYPHER_FORMS_DEFAULT_CLEANUP,
    CYPHER_UPSERT_COMMAND,
    CYPHER_UPSERT_LAYOUT,
    CYPHER_UPSERT_CHARACTERISTIC,
    CYPHER_UPSERT_ENUM_VALUE,
    CYPHER_UPSERT_URL_TEMPLATE,
    CYPHER_UPSERT_URL_METHOD,
    CYPHER_UPSERT_JOURNAL_GRAPH,
    CYPHER_UPSERT_DO_MOVEMENTS_IN,
    CYPHER_UPSERT_ACCOUNTING_FLAG,
    CYPHER_UPSERT_DIMENSION_ACCOUNTING_FLAG,
    CYPHER_UPSERT_TABULAR_PART,
    CYPHER_UPSERT_OBJECT_ATTRIBUTE,
    CYPHER_UPSERT_RESOURCE,
    CYPHER_UPSERT_DIMENSION,
    CYPHER_UPSERT_TABULAR_ATTRIBUTE,
    cypher_used_in,
)

logger = logging.getLogger(__name__)


class ConfigLoaderMixin:
    """Chunked loading of Configuration and all nested entities"""

    def repair_subsystem_category_edges(self, project_name: str) -> int:
        """Ребро MetadataCategory «Подсистемы» → подсистема для вложенных подсистем, загруженных до этой правки.

        2.0.1 связывало с категорией только подсистемы верхнего уровня, вложенные висели под родителем,
        и инструменты по категории (get_metadata objects, find_metadata_objects) их не видели. MERGE
        идемпотентный: после первого прохода возвращает 0. Вызывается из PostLinkingSync каждый цикл.
        """
        with self.driver.session(database=settings.neo4j_database) as session:
            rec = session.run(CYPHER_REPAIR_SUBSYSTEM_CATEGORY_EDGES, project_name=project_name).single()
            return int(rec["created"]) if rec else 0

    def load_configurations(
        self,
        configurations: List[Configuration],
        *,
        ensure_indexes: bool = True,
        is_extension: bool = False,
        use_startup_probe_for_vectors: bool = False,
    ) -> None:
        """Load multiple configurations into Neo4j.

        `ensure_indexes` (default True) запускает `create_indexes()` для совместимости
        с существующими call sites. Incremental pipeline вызывает `create_indexes()`
        самостоятельно один раз перед серией bulk apply и передаёт `ensure_indexes=False`,
        чтобы не повторять "Database indexes ensured" / "Ensuring vector indexes" на
        каждом цикле.

        `is_extension` (default False) различает базовую и расширенную загрузку — флаг
        прокидывается в `_load_configuration`, который пишет `Configuration.is_extension`.
        Любой apply внутри extension scope обязан идти с `is_extension=True`, включая
        Configuration-only diff (иначе incremental Configuration update перезапишет
        `is_extension=false`, что ломает поиск базовой конфигурации в mcpsrv/server.py).
        """
        if not configurations:
            logger.info("No configurations to load")
            return

        if ensure_indexes:
            try:
                self.create_indexes(  # from IndexManagementMixin
                    use_startup_probe_for_vectors=use_startup_probe_for_vectors
                )
            except Exception as e:
                logger.warning("create_indexes failed or partially applied: %s", e)

        with self.driver.session(database=settings.neo4j_database) as session:
            project_name = settings.project_name

            # Create or get the Project node (by configured project name)
            project_result = session.run(CYPHER_MERGE_PROJECT, project_name=project_name)
            _ = project_result.single()
            logger.info("Created/found Project node: %s", project_name)

            # Load each configuration
            for config in configurations:
                self._load_configuration(session, project_name, config, is_extension=is_extension)

        logger.info("Successfully loaded %d configurations", len(configurations))

    def _load_configuration(
        self,
        session,
        project_name: str,
        config: Configuration,
        is_extension: bool = False
    ) -> None:
        """Load a single configuration using many small write transactions (commit per chunk)."""
        logger.info("Loading configuration (chunked): %s (is_extension=%s)",
                    config.name, is_extension)

        # 0) Upsert Configuration node and link to Project (single small tx)
        def _tx_upsert_cfg(tx, payload: Dict[str, Any]):
            tx.run(CYPHER_UPSERT_CONFIGURATION, **payload)

        config_qn = f"{project_name}/{config.name}"
        self._write(session, _tx_upsert_cfg, {
            "project_name": project_name,
            "config_name": config.name,
            "config_qn": config_qn,
            "is_extension": is_extension,
            "properties": config.properties,
        })

        # 1) Build all row collections outside of any tx
        rows: Dict[str, Any] = self._build_rows_for_configuration(
            project_name, config
        )
        bs = settings.neo4j_batch_size

        # Helper to run a single UNWIND statement for a chunk
        def _tx_unwind(tx, cypher: str, chunk_rows: list, extra_params: dict | None = None):
            params = {"rows": chunk_rows}
            if extra_params:
                params.update(extra_params)
            tx.run(cypher, **params)

        # 2) Categories (usually small, single tx)
        if rows.get("categories"):
            self._write(session, _tx_unwind, CYPHER_UPSERT_CATEGORIES, rows["categories"], {"project_name": project_name})

        # 3) Objects (MetadataObject)
        for chunk in self._chunked(rows.get("objects") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_METADATA_OBJECT, chunk)

        # 3h) Subsystem hierarchy
        for chunk in self._chunked(rows.get("subsystem_edges") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_SUBSYSTEM_EDGE, chunk)

        # 3b) Forms
        for chunk in self._chunked(rows.get("forms") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_FORM, chunk)

        # 3c) Cleanup defaults
        if rows.get("default_cleanup"):
            self._write(session, _tx_unwind, CYPHER_FORMS_DEFAULT_CLEANUP, rows["default_cleanup"])

        # 3d) Commands
        for chunk in self._chunked(rows.get("commands") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_COMMAND, chunk)

        # 3e) Layouts
        for chunk in self._chunked(rows.get("layouts") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_LAYOUT, chunk)

        # 3f) Characteristics
        for chunk in self._chunked(rows.get("schemes") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_CHARACTERISTIC, chunk)

        # 3g) Enum values
        for chunk in self._chunked(rows.get("enum_vals") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_ENUM_VALUE, chunk)

        # 3h) UrlTemplates
        for chunk in self._chunked(rows.get("url_templates") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_URL_TEMPLATE, chunk)

        # 3i) UrlMethods
        for chunk in self._chunked(rows.get("url_methods") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_URL_METHOD, chunk)

        # 3j) JournalGraphs
        for chunk in self._chunked(rows.get("journal_graphs") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_JOURNAL_GRAPH, chunk)

        # 3m) Document movements
        for chunk in self._chunked(rows.get("movements") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_DO_MOVEMENTS_IN, chunk)

        # 3k) Accounting Flags
        for chunk in self._chunked(rows.get("account_flags") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_ACCOUNTING_FLAG, chunk)

        # 3l) Subconto Accounting Flags
        for chunk in self._chunked(rows.get("subconto_flags") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_DIMENSION_ACCOUNTING_FLAG, chunk)

        # 4) TabularParts
        for chunk in self._chunked(rows.get("tabulars") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_TABULAR_PART, chunk)

        # 5) Object-level Attributes
        for chunk in self._chunked(rows.get("obj_attrs") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_OBJECT_ATTRIBUTE, chunk)

        # 6) Resources
        for chunk in self._chunked(rows.get("resources") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_RESOURCE, chunk)

        # 7) Dimensions
        for chunk in self._chunked(rows.get("dimensions") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_DIMENSION, chunk)

        # 8) TabularPart Attributes
        for chunk in self._chunked(rows.get("tab_attrs") or [], bs):
            self._write(session, _tx_unwind, CYPHER_UPSERT_TABULAR_ATTRIBUTE, chunk)

        # 9) USED_IN relationships
        usage_rows = rows.get("usage_rows") or []
        if usage_rows:
            usage_attr = [r for r in usage_rows if r.get('consumer_label') == 'Attribute']
            usage_res  = [r for r in usage_rows if r.get('consumer_label') == 'Resource']
            usage_dim  = [r for r in usage_rows if r.get('consumer_label') == 'Dimension']
            usage_af   = [r for r in usage_rows if r.get('consumer_label') == 'AccountingFlag']
            usage_scf  = [r for r in usage_rows if r.get('consumer_label') == 'DimensionAccountingFlag']

            for chunk in self._chunked(usage_attr, bs):
                self._write(session, _tx_unwind, cypher_used_in("Attribute"), chunk)
            for chunk in self._chunked(usage_res, bs):
                self._write(session, _tx_unwind, cypher_used_in("Resource"), chunk)
            for chunk in self._chunked(usage_dim, bs):
                self._write(session, _tx_unwind, cypher_used_in("Dimension"), chunk)
            for chunk in self._chunked(usage_af, bs):
                self._write(session, _tx_unwind, cypher_used_in("AccountingFlag"), chunk)
            for chunk in self._chunked(usage_scf, bs):
                self._write(session, _tx_unwind, cypher_used_in("DimensionAccountingFlag"), chunk)

    def create_extends_link(self, ext_qn: str, base_qn: str) -> None:
        """Create EXTENDS relationship between extension and base configuration"""
        from .cypher_templates import CYPHER_CREATE_EXTENDS

        with self.driver.session(database=settings.neo4j_database) as session:
            def _tx_extends(tx):
                tx.run(CYPHER_CREATE_EXTENDS, ext_qn=ext_qn, base_qn=base_qn)

            self._write(session, _tx_extends)
            logger.info("✓ EXTENDS created: %s → %s", ext_qn, base_qn)