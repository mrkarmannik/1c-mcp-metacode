"""
Core Neo4j client: connection management, transactional helpers, and maintenance.
Pulled out of the monolithic loader to be reused by mixins.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional
import logging

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

from config import settings
from .cypher_templates import (
    CYPHER_CLEAR_DATABASE,
)

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Base client that holds the driver and common helpers (sessions/tx/chunking)."""

    def __init__(self) -> None:
        """Initialize Neo4j connection"""
        self.driver: Optional[Driver] = None
        # Optional separate read-only driver can be added later if needed
        self.ro_driver: Optional[Driver] = None
        self.connect()

    def connect(self) -> None:
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                connection_timeout=settings.neo4j_connection_timeout,
                max_connection_lifetime=settings.neo4j_max_conn_lifetime,
                max_connection_pool_size=settings.neo4j_pool_size,
                max_transaction_retry_time=settings.neo4j_max_tx_retry_time,
            )
            # Test connection
            with self.driver.session(database=settings.neo4j_database) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j at %s", settings.neo4j_uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            raise

    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
        if getattr(self, "ro_driver", None):
            try:
                self.ro_driver.close()
            except Exception:
                pass

    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database"""
        if not self.driver:
            raise RuntimeError("Neo4j driver is not initialized")
        with self.driver.session(database=settings.neo4j_database) as session:
            session.run(CYPHER_CLEAR_DATABASE)
            logger.info("Database cleared")

    def clear_project(self, project_name: str) -> None:
        """
        Clear only nodes and relationships that belong to a specific project.

        Strategy:
        - Delete all nodes reachable from the Project node via outgoing paths.
        - Additionally delete nodes with property project_name equal to the target.
        - Additionally delete nodes whose qualified_name starts with "<project_name>/".
        Finally delete the Project node itself.

        Uses batched deletion to avoid memory exhaustion on large projects (millions of nodes).
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver is not initialized")

        batch_size = settings.neo4j_clear_project_batch_size
        total_deleted = 0

        with self.driver.session(database=settings.neo4j_database) as session:
            # Strategies 1-2 (по project_name и по префиксу qualified_name) — по каждой
            # метке графа. Исходный label-free `MATCH (n) ... LIMIT` просматривал
            # весь общий граф на КАЖДЫЙ батч (O(узлы графа × батчи)); с меткой Neo4j
            # берёт индексы qualified_name (уникальные у metadata-меток) и project_name
            # (Module/Routine/MetadataObject), а метки без них сканирует в пределах
            # метки. Условия принадлежности те же. Узел Project удаляется последним.
            prefix = project_name + '/'
            labels = [
                row[0] for row in session.run("CALL db.labels() YIELD label RETURN label").values()
                if row[0] != "Project"
            ]
            # Ветки по метке выбираются по её индексам (clear_project_branches). Метка с
            # индексом project_name и без индекса qualified_name (ConsoleSearchable, 6 млн узлов)
            # чистится только по project_name: ветка STARTS WITH там была полным просмотром метки
            # и ничего не находила — узлы уже удалены через основные метки.
            qn_indexed, pn_indexed = _label_index_sets(session)
            logger.info("Clearing project nodes by project_name / qualified_name prefix (%d labels)...",
                        len(labels))
            for label in labels:
                escaped = "`" + label.replace("`", "``") + "`"
                for where in clear_project_branches(label, qn_indexed, pn_indexed):
                    label_deleted = 0
                    while True:
                        result = session.run(
                            f"MATCH (n:{escaped}) WHERE {where} "
                            "WITH n LIMIT $batch_size DETACH DELETE n RETURN count(n) AS deleted",
                            project_name=project_name, prefix=prefix, batch_size=batch_size,
                        )
                        record = result.single()
                        deleted = record['deleted'] if record else 0
                        total_deleted += deleted
                        label_deleted += deleted
                        if deleted < batch_size:
                            break
                    if label_deleted:
                        logger.info("  %s: deleted %d nodes (total: %d)", label, label_deleted, total_deleted)

            # Strategy 3: Delete nodes reachable from Project via graph traversal
            logger.info("Clearing project nodes by graph traversal...")
            while True:
                result = session.run("""
                    MATCH (p:Project {name: $project_name})-[*1..]->(n)
                    WITH n LIMIT $batch_size
                    DETACH DELETE n
                    RETURN count(n) AS deleted
                    """,
                    project_name=project_name,
                    batch_size=batch_size
                )

                record = result.single()
                deleted = record['deleted'] if record else 0
                total_deleted += deleted

                if deleted > 0:
                    logger.info("  Deleted %d nodes (total: %d)", deleted, total_deleted)

                if deleted < batch_size:
                    break

            # Finally: Delete the Project node itself
            session.run("MATCH (p:Project {name: $project_name}) DETACH DELETE p", project_name=project_name)

            logger.info("Project cleared: %s (total nodes deleted: %d)", project_name, total_deleted)

    def _write(self, session, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Compatibility wrapper for transaction functions across driver versions."""
        try:
            return session.execute_write(func, *args, **kwargs)  # Neo4j Python 5+
        except AttributeError:
            return session.write_transaction(func, *args, **kwargs)  # Neo4j Python 4.x

    def _chunked(self, rows: Optional[Iterable[Any]], size: int = None):
        """Yield list chunks of a given size."""
        if not rows:
            return
        if size is None:
            size = settings.neo4j_batch_size
        rows_list = list(rows)
        for i in range(0, len(rows_list), size):
            yield rows_list[i : i + size]

    def _get_read_driver(self) -> Driver:
        """Return a driver for read-only operations. Currently reuses main driver."""
        # Placeholder for future separate RO credentials/driver if needed
        if not self.driver:
            raise RuntimeError("Neo4j driver is not initialized")
        return self.driver


CLEAR_PROJECT_BRANCHES = ("n.project_name = $project_name", "n.qualified_name STARTS WITH $prefix")


def clear_project_branches(label: str, qn_indexed: set, pn_indexed: set) -> tuple:
    """Условия удаления узлов метки в clear_project.

    Обе ветки — для меток с индексом по qualified_name (STARTS WITH идёт по индексу) и для меток
    без индексов (маленькие служебные метки, полный просмотр дёшев). Только project_name — для
    меток с индексом по project_name без индекса qualified_name: это дублирующие метки вроде
    ConsoleSearchable, их узлы уже удалены через основные метки, а STARTS WITH без индекса —
    просмотр всей метки (6 млн узлов, секунды на каждом проекте и при каждой переиндексации).
    """
    if label in pn_indexed and label not in qn_indexed:
        return CLEAR_PROJECT_BRANCHES[:1]
    return CLEAR_PROJECT_BRANCHES


def _label_index_sets(session) -> tuple:
    """Метки с одиночным индексом по qualified_name и по project_name.

    Только одиночные индексы: равенство по одному свойству планировщик по составному индексу
    (`ConsoleSearchable(project_name, console_search_section)`) не ищет — остаётся просмотр метки.
    """
    qn_indexed, pn_indexed = set(), set()
    rows = session.run(
        "SHOW INDEXES YIELD labelsOrTypes, properties, entityType, type "
        "WHERE entityType = 'NODE' AND type <> 'FULLTEXT' RETURN labelsOrTypes, properties"
    ).values()
    for labels, props in rows:
        for lbl in labels or []:
            if props == ["qualified_name"]:
                qn_indexed.add(lbl)
            if props == ["project_name"]:
                pn_indexed.add(lbl)
    return qn_indexed, pn_indexed
