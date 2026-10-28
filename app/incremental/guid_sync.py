"""
GuidIncrementalSync: top-level GUID sync для incremental loading.

- Phase 0 в sync_txt/sync_xml: запускается до metadata fast-path.
- Соблюдает settings.load_metadata_guids guard (parity с full load).
- Scoped refresh meta_uuid когда ConfigDumpInfo.xml изменился, но metadata не парсится.
- Lifecycle: guid_state row owned by IncrementalLoadingState (delete_scope / reset_after_full_reload).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dumpinfo_loader import load_dumpinfo_map
from .hashing import compute_file_hash
from .state import IncrementalLoadingState

logger = logging.getLogger(__name__)


@dataclass
class GuidSyncOutcome:
    """Результат sync для одного scope.

    map: текущий GUID map (xcf_name -> guid); пустой если guard выключен/файла нет.
    changed: изменился ли ConfigDumpInfo.xml с прошлого прогона; False если enabled=False.
    enabled: settings.load_metadata_guids на момент вызова.
    file_stats: (size, mtime_ns, content_hash) для baseline manifest; None если файла нет.
    """
    map: Dict[str, str] = field(default_factory=dict)
    changed: bool = False
    enabled: bool = True
    file_stats: Optional[Tuple[int, int, str]] = None


# Labels, по которым _enrich_guids ставит meta_uuid (см. graphdb/guid.py).
GUID_REFRESH_LABELS = (
    "MetadataObject",
    "TabularPart",
    "Attribute",
    "Resource",
    "Dimension",
    "Form",
)


def _base_scope() -> str:
    return "guid:base"


def _ext_scope(mode: str, ext_dir: str) -> str:
    return f"guid_ext:{mode}:{ext_dir}"


def _read_guid_map_and_stats(
    code_dir: Path,
) -> Tuple[Dict[str, str], Optional[Tuple[int, int, str]]]:
    """Прочитать ConfigDumpInfo.xml. Вернуть (map, (size, mtime_ns, content_hash))
    или (пустой, None) если файла нет."""
    xml_path = code_dir / "ConfigDumpInfo.xml"
    if not xml_path.exists():
        return {}, None
    try:
        stat = xml_path.stat()
        data = xml_path.read_bytes()
        content_hash = compute_file_hash(data)
    except Exception as e:  # noqa: BLE001
        logger.warning("GUID sync: failed to stat/read %s: %s", xml_path, e)
        return {}, None
    try:
        guid_map = load_dumpinfo_map(code_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("GUID sync: failed to parse %s: %s", xml_path, e)
        guid_map = {}
    return guid_map, (stat.st_size, stat.st_mtime_ns, content_hash)


class GuidIncrementalSync:
    """Phase 0 GUID sync для base и extensions."""

    def apply_for_base(
        self,
        loader: Any,
        settings: Any,
        state: IncrementalLoadingState,
    ) -> GuidSyncOutcome:
        if not getattr(settings, "load_metadata_guids", True):
            return GuidSyncOutcome(enabled=False)

        code_dir = Path(settings.code_directory)
        guid_map, stats = _read_guid_map_and_stats(code_dir)
        scope = _base_scope()
        changed = self._is_file_changed(state, scope, stats)
        return GuidSyncOutcome(
            map=guid_map, changed=changed, enabled=True, file_stats=stats
        )

    def apply_for_extension(
        self,
        loader: Any,
        settings: Any,
        state: IncrementalLoadingState,
        mode: str,
        ext_dir: str,
        code_dir: Path,
    ) -> GuidSyncOutcome:
        if not getattr(settings, "load_metadata_guids", True):
            return GuidSyncOutcome(enabled=False)

        guid_map, stats = _read_guid_map_and_stats(code_dir)
        scope = _ext_scope(mode, ext_dir)
        changed = self._is_file_changed(state, scope, stats)
        return GuidSyncOutcome(
            map=guid_map, changed=changed, enabled=True, file_stats=stats
        )

    @staticmethod
    def _is_file_changed(
        state: IncrementalLoadingState,
        scope: str,
        stats: Optional[Tuple[int, int, str]],
    ) -> bool:
        prev = state.get_source_manifest("guid", scope)
        if stats is None:
            # Файла нет сейчас; считаем изменением только если раньше был.
            return prev is not None
        size, mtime_ns, content_hash = stats
        if prev is None:
            return True
        # Fast-path mtime+size; fallback на content_hash.
        if prev["mtime_ns"] == mtime_ns and prev["size"] == size:
            return False
        return prev["content_hash"] != content_hash

    def scoped_refresh(
        self,
        loader: Any,
        settings: Any,
        state: IncrementalLoadingState,
        scope: str,
        guid_map: Dict[str, str],
        file_stats: Optional[Tuple[int, int, str]],
    ) -> None:
        """Применить refresh meta_uuid на узлах текущего scope-а.

        1. Прочитать guid_state rows → eligible {(label, qn, xcf_name, current_guid)}.
        2. Для каждой row сравнить current_guid vs new_map.get(xcf_name) → SET/REMOVE/skip.
        3. После Neo4j apply — обновить guid_state.current_guid и source_manifest.
        """
        if file_stats is None:
            # ConfigDumpInfo.xml исчез (например, проект перешёл на выгрузку из git без него).
            # meta_uuid на узлах теперь ставит разбор XML (атрибут uuid, см. xml_metadata/parser.py),
            # снимать их нельзя; manifest удаляем, чтобы scope не считался изменившимся каждый цикл.
            state.delete_source_manifest("guid", [scope])
            logger.info(
                "GUID scoped refresh: scope=%s — ConfigDumpInfo.xml отсутствует, "
                "GUID берутся из атрибутов uuid в XML объектов", scope,
            )
            return

        eligible = state.get_guid_state_for_scope(scope)
        if not eligible:
            # Если baseline ещё не записан (первый incremental без full reload baseline),
            # делать ничего невозможно — нет regiseterd nodes.
            self._update_manifest(state, scope, file_stats)
            return

        # diff. Только SET: записи, которых в ConfigDumpInfo.xml нет, не повод снимать meta_uuid —
        # его мог поставить разбор XML (атрибут uuid), а узлы удалённых объектов убирает metadata sync.
        set_rows: Dict[str, List[Dict[str, str]]] = {}  # label -> [{qn, meta_uuid}]
        updates: List[Tuple[str, str, Optional[str]]] = []  # (label, qn, new_guid)
        absent = 0

        for row in eligible:
            label = row["label"]
            qn = row["qualified_name"]
            xcf = row["xcf_name"]
            current = row.get("current_guid")
            new_guid = guid_map.get(xcf)
            if current == new_guid:
                continue
            if new_guid is None:
                absent += 1
                continue
            # added or changed
            set_rows.setdefault(label, []).append({"qn": qn, "meta_uuid": new_guid})
            updates.append((label, qn, new_guid))

        if absent:
            logger.debug("GUID scoped refresh: scope=%s — %d узлов без записи в ConfigDumpInfo.xml, "
                         "meta_uuid оставлен", scope, absent)
        if not set_rows:
            self._update_manifest(state, scope, file_stats)
            return

        try:
            session_db = getattr(settings, "neo4j_database", None)
            with loader.driver.session(database=session_db) as session:
                for label, rows in set_rows.items():
                    session.run(
                        f"UNWIND $rows AS row "
                        f"MATCH (n:`{label}` {{qualified_name: row.qn, "
                        f"project_name: $project_name}}) "
                        f"SET n.meta_uuid = row.meta_uuid",
                        rows=rows,
                        project_name=settings.project_name,
                    )
        except Exception as e:  # noqa: BLE001
            logger.error("GUID scoped refresh: Neo4j apply failed for scope=%s: %s",
                         scope, e, exc_info=True)
            return  # не обновляем sidecar — следующий прогон попробует ещё раз

        # commit sidecar
        with state.transaction():
            state.update_guid_state_current_guids(scope, updates)
            self._update_manifest(state, scope, file_stats)

        logger.info(
            "GUID scoped refresh: scope=%s set=%d absent_in_dumpinfo=%d",
            scope,
            sum(len(v) for v in set_rows.values()),
            absent,
        )

    @staticmethod
    def _update_manifest(
        state: IncrementalLoadingState,
        scope: str,
        file_stats: Optional[Tuple[int, int, str]],
    ) -> None:
        if file_stats is None:
            return
        size, mtime_ns, content_hash = file_stats
        state.upsert_source_manifest(
            source_type="guid",
            rel_path=scope,
            size=size,
            mtime_ns=mtime_ns,
            content_hash=content_hash,
        )
