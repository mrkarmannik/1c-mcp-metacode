"""
Read-only data builders for the web console analysis page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import settings
from mcpsrv.neo4j_init import get_loader
from mcpsrv.resolvers import QN_LABEL_EXPR


DEFAULT_LIMIT = 100
MAX_LIMIT = 500
SEARCH_DEFAULT_LIMIT = 50
SEARCH_MAX_LIMIT = 100
SEARCH_TYPE_DEFAULTS = [
    "objects",
    "attributes",
    "standard_attributes",
    "tabular_parts",
    "tabular_part_attributes",
    "resources",
    "dimensions",
    "forms",
    "commands",
    "layouts",
    "journal_graphs",
    "enum_values",
    "predefined",
    "modules",
    "form_attributes",
    "form_controls",
]
SEARCH_FIELD_DEFAULTS = [
    "name",
    "synonym",
    "comment",
    "type",
    "category",
    "config",
    "path",
]

TECHNICAL_PROPERTY_KEYS = {
    "qualified_name",
    "name",
    "Name",
    "project_name",
    "config_name",
    "category_name",
    "object_name",
    "tabular_name",
    "meta_uuid",
    "uuid",
    "owner_qn",
    "owner_name",
    "owner_kind",
    "id",
    "path",
    "file_path",
    "line",
    "ctrl_id",
    "base_control_id",
    "content_hash",
    "form_content_hash",
    "base_form_hash",
    "body_hash",
    "body",
    "body_sample",
    "params_json_str",
    "description_embedding",
    "doc_description_embedding",
    "object_summary_embedding",
    "object_summary_search_text",
    "object_summary_path",
    "code_embedding",
    "code_embedding_epoch",
    "code_embedding_visible",
    "ПутьКДанным_RAW",
}

EXTENSION_PROPERTY_KEYS = {
    "ext_source",
    "modified_properties",
    "modified_values",
    "controlled_properties",
    "controlled_values",
}

PROPERTY_LABELS = {
    "name_path": "Путь элемента",
}

PROPERTY_GROUPS = {
    "identity": {
        "title": "Идентификация",
        "keys": ["Имя", "Синоним", "Тип", "Комментарий", "Пояснение", "Описание", "Справка", "Подсказка"],
    },
    "forms": {
        "title": "Формы",
        "keys": [
            "ОсновнаяФорма",
            "ДополнительнаяФорма",
            "ОсновнаяФормаОбъекта",
            "ДополнительнаяФормаОбъекта",
            "ОсновнаяФормаСписка",
            "ДополнительнаяФормаСписка",
            "ОсновнаяФормаДляВыбора",
            "ДополнительнаяФормаДляВыбора",
            "ОсновнаяФормаГруппы",
            "ДополнительнаяФормаГруппы",
            "ОсновнаяФормаДляВыбораГруппы",
            "ДополнительнаяФормаДляВыбораГруппы",
        ],
    },
    "posting": {
        "title": "Проведение и движения",
        "keys": [
            "Проведение",
            "Движения",
            "ЗаписьДвиженийПриПроведении",
            "УдалениеДвижений",
            "ОперативноеПроведение",
            "ПривилегированныйРежимПриПроведении",
            "ПривилегированныйРежимПриОтменеПроведения",
        ],
    },
    "numbering": {
        "title": "Нумерация и иерархия",
        "keys": [
            "Автонумерация",
            "ТипНомера",
            "ДлинаНомера",
            "ДопустимаяДлинаНомера",
            "ПериодичностьНомера",
            "ТипКода",
            "ДлинаКода",
            "ДопустимаяДлинаКода",
            "ДлинаНаименования",
            "Иерархический",
            "ВидИерархии",
            "КоличествоУровней",
            "ОграничиватьКоличествоУровней",
            "ГруппыСверху",
            "Владельцы",
            "ИспользованиеПодчинения",
        ],
    },
    "module": {
        "title": "Модульность",
        "keys": [
            "Сервер",
            "ВызовСервера",
            "КлиентУправляемоеПриложение",
            "КлиентОбычноеПриложение",
            "ВнешнееСоединение",
            "Привилегированный",
            "Глобальный",
            "ПовторноеИспользованиеВозвращаемыхЗначений",
        ],
    },
}


def _loader():
    loader = get_loader()
    if loader is None:
        raise RuntimeError("Neo4j database connection not available")
    return loader


def _run(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return _loader().execute_query_readonly(cypher, params or {}) or []


def _limit(value: Any, default: int = DEFAULT_LIMIT) -> int:
    try:
        return max(1, min(MAX_LIMIT, int(value if value is not None else default)))
    except (TypeError, ValueError):
        return default


def _offset(value: Any) -> int:
    try:
        return max(0, int(value if value is not None else 0))
    except (TypeError, ValueError):
        return 0


def _search_limit(value: Any) -> int:
    try:
        return max(1, min(SEARCH_MAX_LIMIT, int(value if value is not None else SEARCH_DEFAULT_LIMIT)))
    except (TypeError, ValueError):
        return SEARCH_DEFAULT_LIMIT


def _search_needle(value: Any) -> tuple[str, str]:
    query = str(value or "").strip()
    normalized = (
        query.lower()
        .replace("ё", "е")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )
    return query.lower(), normalized


# Lucene special characters that need escaping inside fulltext query strings.
# We DON'T escape '*' because we intentionally append it as a prefix wildcard.
_LUCENE_SPECIAL = r'+-&|!(){}[]^"~:\/'


def _lucene_escape(token: str) -> str:
    out = []
    for ch in token:
        if ch in _LUCENE_SPECIAL:
            out.append("\\")
        out.append(ch)
    return "".join(out)


# Map UI field name -> Neo4j indexed property name (Lucene field).
_FULLTEXT_FIELD_MAP = {
    "name": "console_search_name",
    "synonym": "console_search_synonym",
    "type": "console_search_type",
}


def _build_fulltext_lucene(query_text: str, selected_fields: list[str]) -> str:
    """Build Lucene query targeting only those `console_search_*` properties
    that correspond to UI fields the caller actually selected.

    Returns an empty string when no fulltext-applicable field is in `selected_fields`
    or when the query has no usable tokens — caller should then omit the fulltext branch.
    """
    properties = [
        _FULLTEXT_FIELD_MAP[f] for f in selected_fields if f in _FULLTEXT_FIELD_MAP
    ]
    if not properties:
        return ""

    raw_tokens = [t for t in re.split(r"\s+", query_text or "") if t]
    if not raw_tokens:
        return ""

    clauses: list[str] = []
    for token in raw_tokens:
        esc = _lucene_escape(token)
        for prop in properties:
            # prefix match + fuzzy edit-distance 1 — covers typos and partial input
            clauses.append(f"{prop}:{esc}*")
            clauses.append(f"{prop}:{esc}~1")
    return " OR ".join(clauses)


def _csv_scope(value: Any, defaults: list[str]) -> list[str]:
    if value is None:
        return defaults
    items = [
        item.strip()
        for item in str(value or "").split(",")
        if item.strip()
    ]
    allowed = set(defaults)
    filtered = [item for item in items if item in allowed]
    return filtered


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def _is_denied_property(key: str) -> bool:
    lower = key.lower()
    return (
        key in TECHNICAL_PROPERTY_KEYS
        or key in EXTENSION_PROPERTY_KEYS
        or "embedding" in lower
        or lower.startswith("object_summary_")
        or lower.endswith("_hash")
        or "content_hash" in lower
        or "fingerprint" in lower
    )


def _safe_props(props: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sorted((props or {}).items(), key=lambda item: item[0].lower()):
        if _is_denied_property(key) or _is_empty(value):
            continue
        out[key] = value
    return out


def _object_summary(
    props: dict[str, Any], *, qualified_name: str
) -> dict[str, Any]:
    path_value = (props or {}).get("object_summary_path")
    available = False
    human_summary: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    if path_value:
        summary_path = Path(str(path_value))
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("human_summary"), dict):
            human = payload["human_summary"]
            human_summary = {
                "title": str(human.get("title") or ""),
                "core_idea": str(human.get("core_idea") or ""),
                "data_scope": str(human.get("data_scope") or ""),
                "capabilities": [
                    {
                        "title": str(item.get("title") or ""),
                        "description": str(item.get("description") or ""),
                    }
                    for item in (human.get("capabilities") or [])
                    if isinstance(item, dict)
                ],
                "usage_scenarios": [
                    {
                        "title": str(item.get("title") or ""),
                        "description": str(item.get("description") or ""),
                    }
                    for item in (human.get("usage_scenarios") or [])
                    if isinstance(item, dict)
                ],
                "effects": str(human.get("effects") or ""),
                "uncertainties": str(human.get("uncertainties") or ""),
            }
            meta = _object_summary_meta(summary_path)
            available = True
    action_state = build_action_state(
        qualified_name=qualified_name,
        props=props or {},
        has_summary=available,
    )
    block: dict[str, Any] = {
        "available": available,
        "action_state": action_state,
    }
    if available:
        block["human_summary"] = human_summary
        block["meta"] = meta
    return block


def build_action_state(
    *, qualified_name: str, props: dict[str, Any], has_summary: bool,
) -> dict[str, Any]:
    """Compute action_state for the web console summary block.

    Wraps environment, runtime readiness and eligibility into the single
    payload consumed by `renderObjectSummaryBlock` and by the dedicated
    status endpoint. See plan §6.
    """
    enabled = bool(getattr(settings, "object_summary_enabled", False))
    regeneration_enabled = bool(
        getattr(settings, "object_summary_manual_regeneration_enabled", True)
    )

    from object_summary.manual_jobs import get_manual_job_manager
    from mcpsrv import runtime_state

    active_job = get_manual_job_manager().get_active()
    active_snapshot = active_job.snapshot() if active_job is not None else None
    startup_ready = runtime_state.is_ready()

    if not enabled:
        return {
            "enabled": False,
            "startup_ready": startup_ready,
            "eligible": False,
            "has_summary": has_summary,
            "regeneration_enabled": regeneration_enabled,
            "can_create": False,
            "can_update": False,
            "disabled_reason": "feature_disabled",
            "active_job": active_snapshot,
        }

    eligibility = _check_manual_eligibility(qualified_name=qualified_name, props=props)
    eligible = eligibility[0]
    disabled_reason = eligibility[1] if not eligible else ""

    can_create = enabled and eligible and (not has_summary) and startup_ready
    can_update = (
        enabled and eligible and has_summary and regeneration_enabled and startup_ready
    )
    if not eligible:
        # disabled_reason already set above
        pass
    elif not startup_ready:
        disabled_reason = "startup_not_ready"
    elif has_summary and not regeneration_enabled:
        disabled_reason = "regeneration_disabled"

    return {
        "enabled": True,
        "startup_ready": startup_ready,
        "eligible": eligible,
        "has_summary": has_summary,
        "regeneration_enabled": regeneration_enabled,
        "can_create": can_create,
        "can_update": can_update,
        "disabled_reason": disabled_reason,
        "active_job": active_snapshot,
    }


def _check_manual_eligibility(
    *, qualified_name: str, props: dict[str, Any],
) -> tuple[bool, str]:
    """Return (eligible, disabled_reason) for the manual summary action.

    Mirrors the category/extension filter used by startup S1 — same
    `filter_supported_categories` intersection, same extension rules.
    """
    from object_summary.constants import filter_supported_categories

    category = str(props.get("category_name") or "").strip()
    config_name = str(props.get("config_name") or "")
    if not category:
        return False, "category_unknown"

    allowed = set(filter_supported_categories(
        list(getattr(settings, "object_summary_categories", []) or [])
    ))
    if category not in allowed:
        return False, "category_not_eligible"

    is_extension = "$ext$" in config_name
    if not is_extension:
        return True, ""

    if not getattr(settings, "object_summary_generate_for_extensions", False):
        return False, "extensions_disabled"

    ext_names = list(getattr(settings, "object_summary_extension_names", []) or []) or ["*"]
    if ext_names != ["*"] and config_name not in ext_names:
        return False, "extension_not_in_scope"

    scope = (getattr(settings, "object_summary_extension_object_scope", "own") or "own").lower()
    if scope == "own":
        loader = get_loader()
        if loader is None:
            return False, "neo4j_unavailable"
        try:
            from graphdb.object_summary_queries import is_object_own_in_extension
            if not is_object_own_in_extension(
                loader.driver,
                project_name=settings.project_name,
                qualified_name=qualified_name,
            ):
                return False, "extension_object_not_own"
        except Exception:
            return False, "neo4j_unavailable"

    return True, ""


def _object_summary_meta(summary_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads((summary_path.parent / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    usage = payload.get("llm_usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "generated_at": str(payload.get("generated_at") or ""),
        "model": str(payload.get("llm_model") or ""),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_amount": usage.get("cost_amount"),
        "cost_unit": str(usage.get("cost_unit") or ""),
        "cost_source": str(usage.get("cost_source") or ""),
    }


def _group_props(props: dict[str, Any]) -> list[dict[str, Any]]:
    visible = _safe_props(props)
    used: set[str] = set()
    groups: list[dict[str, Any]] = []

    for group in PROPERTY_GROUPS.values():
        items = [
            {"key": _property_label(key), "value": visible[key]}
            for key in group["keys"]
            if key in visible
        ]
        if items:
            used.update(item["key"] for item in items)
            groups.append({"title": group["title"], "items": items})

    other = [
        {"key": _property_label(key), "value": value}
        for key, value in visible.items()
        if key not in used
    ]
    if other:
        groups.append({"title": "Прочие свойства", "items": other})
    return groups


def _property_label(key: str) -> str:
    return PROPERTY_LABELS.get(key, key)


def _extension_badges(
    props: dict[str, Any],
    node_label: str = "",
    node_name: str = "",
) -> list[dict[str, str]]:
    badges: list[dict[str, str]] = []
    seen: set[str] = set()
    effective = _effective_extension_props(props, node_label, node_name)

    def add(label: str, kind: str, title: str = "") -> None:
        key = f"{kind}:{label}"
        if key in seen:
            return
        seen.add(key)
        badges.append({"label": label, "kind": kind, "title": title or label})

    modified_title = _extension_props_title(
        "Изменено",
        effective.get("modified_properties") or [],
        effective.get("modified_values") or [],
    )
    controlled_title = _extension_props_title(
        "Контролируется",
        effective.get("controlled_properties") or [],
        effective.get("controlled_values") or [],
    )

    ext_source = str(effective.get("ext_source") or "").strip()
    if ext_source == "own":
        add("добавлено", "own", "Добавлено в расширении")
    elif ext_source == "adopted_modified" and effective.get("modified_properties"):
        add("изменено", "modified", modified_title)
    elif ext_source == "adopted_modified":
        add("заимствовано", "adopted", "Заимствовано из основной конфигурации")
    elif ext_source in {"adopted", "adopted_unchanged"}:
        add("заимствовано", "adopted", "Заимствовано из основной конфигурации")

    if effective.get("modified_properties"):
        add("изменено", "modified", modified_title)
    if effective.get("controlled_properties"):
        add("контроль", "controlled", controlled_title)
    return badges


def _extension_props_title(title: str, props: list[Any], diffs: list[Any]) -> str:
    names = [str(prop) for prop in props if prop]
    if not names:
        return title
    diff_by_prop = {
        str(diff.get("property")): diff
        for diff in diffs
        if isinstance(diff, dict) and diff.get("property")
    }
    lines = [title + ":"]
    for name in names:
        lines.append(name)
        diff = diff_by_prop.get(name)
        if diff and (diff.get("has_base") or diff.get("has_extension")):
            lines.append(f"  База: {_stringify_diff_value(diff.get('base')) if diff.get('has_base') else ''}")
            lines.append(f"  Расширение: {_stringify_diff_value(diff.get('extension')) if diff.get('has_extension') else ''}")
    return "\n".join(lines)


def _effective_extension_props(
    props: dict[str, Any],
    node_label: str = "",
    node_name: str = "",
) -> dict[str, Any]:
    effective = dict(props)
    if node_label not in {"FormAttribute", "FormControl"}:
        return effective

    item = {
        "label": node_label,
        "name": node_name or props.get("name") or props.get("Имя") or "",
        "path": props.get("qualified_name") or "",
        "modified": list(props.get("modified_properties") or []),
        "modified_values": list(props.get("modified_values") or []),
    }
    _suppress_default_form_attribute_title_change(item)
    effective["modified_properties"] = item["modified"]
    effective["modified_values"] = item["modified_values"]
    return effective


def _node_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "name": row.get("name") or "",
        "qualified_name": row.get("qualified_name") or "",
        "category": row.get("category") or "",
        "config_name": row.get("config_name") or "",
    }


_NODE_REF_SEGMENT_REPLACEMENTS = (
    ("/Реквизиты/", "/Attribute/"),
    ("/ТабличныеЧасти/", "/TabularPart/"),
    ("/Ресурсы/", "/Resource/"),
    ("/Измерения/", "/Dimension/"),
)


def _normalize_node_ref(ref: str) -> str:
    normalized = str(ref or "")
    for source, target in _NODE_REF_SEGMENT_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return normalized


def _subsystem_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    path = item.get("subsystem_path") or []
    if isinstance(path, list) and path:
        return tuple(str(part).lower() for part in path)
    return (str(item.get("name") or "").lower(),)


def _nest_subsystems(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_qn: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        item["children"] = []
        qn = item.get("qualified_name")
        if qn:
            by_qn[str(qn)] = item

    roots: list[dict[str, Any]] = []
    for item in by_qn.values():
        parent_qn = item.get("parent_qualified_name")
        parent = by_qn.get(str(parent_qn)) if parent_qn else None
        if parent is None:
            roots.append(item)
            continue
        parent.setdefault("children", []).append(item)

    def sort_branch(items: list[dict[str, Any]]) -> None:
        items.sort(key=_subsystem_sort_key)
        for child in items:
            sort_branch(child.get("children") or [])

    sort_branch(roots)
    return roots


def get_tree() -> dict[str, Any]:
    rows = _run(
        """
        MATCH (p:Project {name: $project_name})-[:HAS_CONFIGURATION]->(c:Configuration)
        OPTIONAL MATCH (c)-[:EXTENDS]->(base:Configuration)
        OPTIONAL MATCH (c)-[:HAS_CATEGORY]->(cat:MetadataCategory)
        OPTIONAL MATCH (cat)-[:CONTAINS_OBJECT]->(m:MetadataObject)
        WITH c, base, cat, count(DISTINCT m) AS object_count
        ORDER BY cat.name
        WITH c, base,
             collect({
               name: cat.name,
               qualified_name: cat.qualified_name,
               object_count: object_count
             }) AS categories
        RETURN c.name AS name,
               c.qualified_name AS qualified_name,
               coalesce(c.is_extension, false) AS is_extension,
               base.name AS extends_name,
               base.qualified_name AS extends_qn,
               categories
        ORDER BY is_extension, name
        """,
        {"project_name": settings.project_name},
    )
    subsystem_count_rows = _run(
        """
        MATCH (m:MetadataObject {project_name: $project_name, category_name: 'Подсистемы'})
        RETURN m.config_name AS config_name, count(DISTINCT m) AS total
        """,
        {"project_name": settings.project_name},
    )
    subsystem_counts = {
        row["config_name"]: int(row.get("total") or 0)
        for row in subsystem_count_rows
        if row.get("config_name")
    }
    return {
        "project": {
            "name": settings.project_name,
            "qualified_name": settings.project_name,
        },
        "configurations": [
            {
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "is_extension": bool(row.get("is_extension")),
                "extends": (
                    {
                        "name": row.get("extends_name"),
                        "qualified_name": row.get("extends_qn"),
                    }
                    if row.get("extends_name")
                    else None
                ),
                "categories": [
                    {
                        **cat,
                        "object_count": subsystem_counts.get(row["name"], cat.get("object_count", 0)),
                    } if cat.get("name") == "Подсистемы" else cat
                    for cat in (row.get("categories") or [])
                    if cat.get("name")
                ],
            }
            for row in rows
        ],
    }


def _get_subsystems_category(config_name: str, category_name: str, lim: int, off: int) -> dict[str, Any]:
    params = {
        "project_name": settings.project_name,
        "config_name": config_name,
        "category_name": category_name,
    }
    rows = _run(
        """
        MATCH (m:MetadataObject {
          project_name: $project_name,
          config_name: $config_name,
          category_name: $category_name
        })
        OPTIONAL MATCH (parent:MetadataObject {
          project_name: $project_name,
          config_name: $config_name,
          category_name: $category_name
        })-[:CONTAINS_OBJECT]->(m)
        WITH m, head([qn IN collect(DISTINCT parent.qualified_name) WHERE qn IS NOT NULL]) AS parent_qualified_name
        RETURN m.name AS name,
               m.qualified_name AS qualified_name,
               parent_qualified_name,
               coalesce(m.`ПутьПодсистемы`, []) AS subsystem_path,
               m.category_name AS category,
               m.config_name AS config_name,
               coalesce(m.`Синоним`, '') AS synonym,
               coalesce(m.`Комментарий`, '') AS comment,
               coalesce(m.`ПринадлежностьОбъекта`, '') AS ownership,
               size([(m)-[:ADOPTED_FROM]->(:MetadataObject) | 1]) > 0 AS is_adopted,
               size([(extension)-[:ADOPTED_FROM]->(m) | 1]) AS extension_adoptions,
               [(extension)-[:ADOPTED_FROM]->(m) | extension.config_name] AS extension_names,
               size([(m)-[:HAS_ATTRIBUTE]->(a:Attribute) WHERE coalesce(a.`Стандартный`, false) <> true | 1]) AS attributes,
               size([(m)-[:HAS_ATTRIBUTE]->(a:Attribute) WHERE coalesce(a.`Стандартный`, false) = true | 1]) AS standard_attributes,
               size([(m)-[:HAS_TABULAR_PART]->(:TabularPart) | 1]) AS tabular_parts,
               size([(m)-[:HAS_RESOURCE]->(:Resource) | 1]) AS resources,
               size([(m)-[:HAS_DIMENSION]->(:Dimension) | 1]) AS dimensions,
               size([(m)-[:HAS_GRAPH]->(:JournalGraph) | 1]) AS journal_graphs,
               size([(m)-[:HAS_FORM]->(:Form) | 1]) AS forms,
               size([(m)-[:HAS_COMMAND]->(:Command) | 1]) AS commands,
               size([(m)-[:HAS_LAYOUT]->(:Layout) | 1]) AS layouts,
               size([(m)-[:HAS_ENUM_VALUE]->(:EnumValue) | 1]) AS enum_values,
               size([(m)-[:HAS_PREDEFINED]->(:PredefinedItem) | 1]) AS predefined,
               size([(m)-[:HAS_MODULE]->(:Module) | 1]) AS modules,
               size([(m)-[:DO_MOVEMENTS_IN]->(:MetadataObject) | 1]) AS movements
        ORDER BY subsystem_path, name
        """,
        params,
    )
    return {
        "config_name": config_name,
        "category": category_name,
        "total": len(rows),
        "limit": lim,
        "offset": off,
        "all_loaded": True,
        "items": _nest_subsystems(rows),
    }


def get_category(config_name: str, category_name: str, limit: Any, offset: Any) -> dict[str, Any]:
    lim = _limit(limit)
    off = _offset(offset)
    if category_name == "Подсистемы":
        return _get_subsystems_category(config_name, category_name, lim, off)

    params = {
        "project_name": settings.project_name,
        "config_name": config_name,
        "category_name": category_name,
        "limit": lim,
        "offset": off,
    }
    count_rows = _run(
        """
        MATCH (:Configuration {project_name: $project_name, name: $config_name})
              -[:HAS_CATEGORY]->(:MetadataCategory {name: $category_name})
              -[:CONTAINS_OBJECT]->(m:MetadataObject)
        RETURN count(DISTINCT m) AS total
        """,
        params,
    )
    rows = _run(
        """
        MATCH (:Configuration {project_name: $project_name, name: $config_name})
              -[:HAS_CATEGORY]->(:MetadataCategory {name: $category_name})
              -[:CONTAINS_OBJECT]->(m:MetadataObject)
        RETURN m.name AS name,
               m.qualified_name AS qualified_name,
               m.category_name AS category,
               m.config_name AS config_name,
               coalesce(m.`Синоним`, '') AS synonym,
               coalesce(m.`Комментарий`, '') AS comment,
               coalesce(m.`ПринадлежностьОбъекта`, '') AS ownership,
               size([(m)-[:ADOPTED_FROM]->(:MetadataObject) | 1]) > 0 AS is_adopted,
               size([(extension)-[:ADOPTED_FROM]->(m) | 1]) AS extension_adoptions,
               [(extension)-[:ADOPTED_FROM]->(m) | extension.config_name] AS extension_names,
               size([(m)-[:HAS_ATTRIBUTE]->(a:Attribute) WHERE coalesce(a.`Стандартный`, false) <> true | 1]) AS attributes,
               size([(m)-[:HAS_ATTRIBUTE]->(a:Attribute) WHERE coalesce(a.`Стандартный`, false) = true | 1]) AS standard_attributes,
               size([(m)-[:HAS_TABULAR_PART]->(:TabularPart) | 1]) AS tabular_parts,
               size([(m)-[:HAS_RESOURCE]->(:Resource) | 1]) AS resources,
               size([(m)-[:HAS_DIMENSION]->(:Dimension) | 1]) AS dimensions,
               size([(m)-[:HAS_FORM]->(:Form) | 1]) AS forms,
               size([(m)-[:HAS_COMMAND]->(:Command) | 1]) AS commands,
               size([(m)-[:HAS_LAYOUT]->(:Layout) | 1]) AS layouts,
               size([(m)-[:HAS_GRAPH]->(:JournalGraph) | 1]) AS journal_graphs,
               size([(m)-[:HAS_ENUM_VALUE]->(:EnumValue) | 1]) AS enum_values,
               size([(m)-[:HAS_PREDEFINED]->(:PredefinedItem) | 1]) AS predefined,
               size([(m)-[:HAS_MODULE]->(:Module) | 1]) AS modules,
               size([(m)-[:DO_MOVEMENTS_IN]->(:MetadataObject) | 1]) AS movements
        ORDER BY name
        SKIP $offset LIMIT $limit
        """,
        params,
    )
    return {
        "config_name": config_name,
        "category": category_name,
        "total": int((count_rows[0] or {}).get("total", 0)) if count_rows else 0,
        "limit": lim,
        "offset": off,
        "items": rows,
    }


def get_search(
    query: Any,
    limit: Any,
    offset: Any,
    config: Any = None,
    types: Any = None,
    fields: Any = None,
) -> dict[str, Any]:
    query_text, normalized = _search_needle(query)
    selected_types = _csv_scope(types, SEARCH_TYPE_DEFAULTS)
    selected_fields = _csv_scope(fields, SEARCH_FIELD_DEFAULTS)
    if not normalized:
        return {
            "query": str(query or ""),
            "total": 0,
            "limit": _search_limit(limit),
            "offset": _offset(offset),
            "items": [],
        }
    if not selected_types or not selected_fields:
        return {
            "query": str(query or "").strip(),
            "total": 0,
            "limit": _search_limit(limit),
            "offset": _offset(offset),
            "items": [],
        }

    lim = _search_limit(limit)
    off = _offset(offset)
    params = {
        "project_name": settings.project_name,
        "project_prefix": f"{settings.project_name}/",
        "needle": query_text,
        "needle_norm": normalized,
        "limit": lim,
        "offset": off,
        "config": str(config or "").strip(),
        "types": selected_types,
        "fields": selected_fields,
    }
    # New search uses pre-computed :ConsoleSearchable fields:
    #   console_search_section  - one of SEARCH_TYPE_DEFAULTS
    #   console_search_name/_synonym/_type      - fulltext properties
    #   console_search_name_norm/_synonym_norm/_type_norm  - normalized for fallback CONTAINS

    # Build Lucene query targeting only the fulltext sub-fields the user actually selected.
    fulltext_lucene = _build_fulltext_lucene(query_text, selected_fields)

    branches: list[str] = []

    if fulltext_lucene:
        branches.append(
            "CALL db.index.fulltext.queryNodes('ftx_console_search_text', $lucene_query) "
            "YIELD node AS n, score "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "RETURN n, score"
        )

    if 'path' in selected_fields:
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND toLower(coalesce(n.qualified_name, '')) CONTAINS $needle "
            "RETURN n, 0.0 AS score"
        )

    if 'comment' in selected_fields:
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND toLower(coalesce(n.`Комментарий`, '')) CONTAINS $needle "
            "RETURN n, 0.0 AS score"
        )

    if 'category' in selected_fields:
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND toLower(coalesce(n.category_name, '')) CONTAINS $needle "
            "RETURN n, 0.0 AS score"
        )

    if 'config' in selected_fields:
        # Scope by `$config` stays — the parameter is a scope filter, not optional.
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND toLower(coalesce(n.config_name, '')) CONTAINS $needle "
            "RETURN n, 0.0 AS score"
        )

    # Fallback for "joined" 1C-style names — only over the *_norm fields whose
    # corresponding fulltext field is selected. Each norm-branch is independent
    # to preserve per-`fields` semantics.
    if 'name' in selected_fields:
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND n.console_search_name_norm CONTAINS $needle_norm "
            "RETURN n, 0.0 AS score"
        )
    if 'synonym' in selected_fields:
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND n.console_search_synonym_norm CONTAINS $needle_norm "
            "RETURN n, 0.0 AS score"
        )
    if 'type' in selected_fields:
        branches.append(
            "MATCH (n:ConsoleSearchable) "
            "WHERE n.project_name = $project_name "
            "AND ($config = '' OR n.config_name = $config) "
            "AND n.console_search_section IN $types "
            "AND n.console_search_type_norm CONTAINS $needle_norm "
            "RETURN n, 0.0 AS score"
        )

    if not branches:
        return {
            "query": str(query or "").strip(),
            "total": 0,
            "limit": lim,
            "offset": off,
            "items": [],
        }

    params["lucene_query"] = fulltext_lucene or ""
    union_block = "\nUNION\n".join(branches)
    cypher = (
        "CALL {\n"
        + union_block
        + "\n}\n"
        "WITH n, max(score) AS score\n"
        "WITH n, score,\n"
        "     CASE\n"
        "       WHEN n.console_search_section = 'objects' THEN 'object'\n"
        "       ELSE 'node'\n"
        "     END AS kind,\n"
        "     toString(coalesce(n.qualified_name, '')) AS ref,\n"
        "     toString(coalesce(n.name, n.`Имя`, n.module_type, 'Модуль')) AS label,\n"
        "     toString(coalesce(n.category_name, '')) AS category,\n"
        "     toString(coalesce(n.config_name, '')) AS config_name,\n"
        "     toString(coalesce(n.console_search_section, '')) AS section,\n"
        "     toString(coalesce(n.`Синоним`, n.`Заголовок`, '')) AS synonym,\n"
        "     toString(coalesce(n.`Комментарий`, '')) AS comment,\n"
        "     coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,\n"
        "     size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,\n"
        "     size([(ext)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,\n"
        "     [(ext)-[:ADOPTED_FROM]->(n) | ext.config_name] AS extension_names\n"
        "WITH kind, ref, label, category, config_name, section, '' AS owner_ref,\n"
        "     synonym, comment,\n"
        "     CASE WHEN kind = 'object' THEN category + '.' + label ELSE ref END AS path,\n"
        "     ownership, is_adopted, extension_adoptions, extension_names, score\n"
        "ORDER BY score DESC, category, label, path\n"
        "WITH collect({\n"
        "  kind: kind,\n"
        "  ref: ref,\n"
        "  label: label,\n"
        "  category: category,\n"
        "  config_name: config_name,\n"
        "  section: section,\n"
        "  owner_ref: owner_ref,\n"
        "  synonym: synonym,\n"
        "  comment: comment,\n"
        "  path: path,\n"
        "  ownership: ownership,\n"
        "  is_adopted: is_adopted,\n"
        "  extension_adoptions: extension_adoptions,\n"
        "  extension_names: extension_names\n"
        "}) AS all_items\n"
        "RETURN size(all_items) AS total, all_items[$offset..($offset + $limit)] AS items\n"
    )
    rows = _run(cypher, params)
    row = rows[0] if rows else {}
    return {
        "query": str(query or "").strip(),
        "total": int(row.get("total") or 0),
        "limit": lim,
        "offset": off,
        "items": row.get("items") or [],
    }


def get_node(ref: str) -> dict[str, Any]:
    ref = _normalize_node_ref(ref)
    rows = _run(
        """
        MATCH (n:Configuration|MetadataCategory|MetadataObject|Form|Command|Attribute|TabularPart|Resource|Dimension|Layout|EnumValue|PredefinedItem|JournalGraph|FormControl|FormAttribute {qualified_name: $ref})
        OPTIONAL MATCH (n)-[:ADOPTED_FROM]->(base_node)
        RETURN labels(n)[0] AS label,
               coalesce(n.name, n.`Имя`, '') AS name,
               n.qualified_name AS qualified_name,
               coalesce(n.category_name, '') AS category,
               coalesce(n.config_name, '') AS config_name,
               properties(n) AS properties,
               [prop IN coalesce(n.modified_properties, []) |
                 {
                   property: prop,
                   base: CASE WHEN base_node IS NOT NULL AND prop IN keys(base_node) THEN base_node[prop] ELSE NULL END,
                   extension: CASE WHEN prop IN keys(n) THEN n[prop] ELSE NULL END,
                   has_base: base_node IS NOT NULL AND prop IN keys(base_node),
                   has_extension: prop IN keys(n)
                 }
               ] AS modified_values,
               [prop IN coalesce(n.controlled_properties, []) |
                 {
                   property: prop,
                   base: CASE WHEN base_node IS NOT NULL AND prop IN keys(base_node) THEN base_node[prop] ELSE NULL END,
                   extension: CASE WHEN prop IN keys(n) THEN n[prop] ELSE NULL END,
                   has_base: base_node IS NOT NULL AND prop IN keys(base_node),
                   has_extension: prop IN keys(n)
                 }
               ] AS controlled_values
        LIMIT 1
        """,
        {"ref": ref},
    )
    if not rows:
        raise ValueError("node_not_found")
    row = rows[0]
    props = dict(row.get("properties") or {})
    props["modified_values"] = row.get("modified_values") or []
    props["controlled_values"] = row.get("controlled_values") or []
    return {
        "node": _node_ref(row),
        "badges": _extension_badges(
            props,
            node_label=row.get("label") or "",
            node_name=row.get("name") or "",
        ),
        "properties": _group_props(props),
        "events": _node_events(ref),
    }


def _node_events(ref: str) -> list[dict[str, Any]]:
    rows = _run(
        """
        MATCH (n:Form|FormControl|MetadataObject {qualified_name: $ref})
        MATCH (n)-[:HAS_EVENT]->(e:FormEvent)
        OPTIONAL MATCH (e)-[:HAS_EVENT_ACTION]->(a:FormEventAction)
        OPTIONAL MATCH (a)-[:HAS_HANDLER]->(r:Routine)
        OPTIONAL MATCH (a)-[:EXTENDS_ACTION]->(base_action:FormEventAction)
        OPTIONAL MATCH (ext_action:FormEventAction)-[:EXTENDS_ACTION]->(a)
        RETURN coalesce(e.name, e.`Имя`, '') AS event_name,
               e.qualified_name AS event_qn,
               coalesce(e.config_name, n.config_name, '') AS config_name,
               a.qualified_name AS action_qn,
               coalesce(a.call_type, '') AS call_type,
               coalesce(a.handler_name, '') AS handler_name,
               coalesce(r.id, '') AS routine_id,
               coalesce(r.name, '') AS routine_name,
               coalesce(r.owner_qn, '') AS routine_owner_qn,
               base_action IS NOT NULL AS extends_base_action,
               count(DISTINCT ext_action) AS extension_action_count
        ORDER BY event_name, call_type, handler_name
        """,
        {"ref": ref},
    )
    by_event: dict[str, dict[str, Any]] = {}
    for row in rows:
        event_name = str(row.get("event_name") or "").strip()
        event_qn = str(row.get("event_qn") or "").strip()
        if not event_name and not event_qn:
            continue
        event_key = event_qn or event_name
        event = by_event.setdefault(
            event_key,
            {
                "name": event_name or event_qn.rsplit("/", 1)[-1],
                "qualified_name": event_qn,
                "config_name": str(row.get("config_name") or ""),
                "actions": [],
            },
        )
        action_qn = str(row.get("action_qn") or "").strip()
        call_type = str(row.get("call_type") or "").strip()
        handler_name = str(row.get("handler_name") or "").strip()
        routine_name = str(row.get("routine_name") or "").strip()
        routine_id = str(row.get("routine_id") or "").strip()
        if not (action_qn or call_type or handler_name or routine_name or routine_id):
            continue
        event["actions"].append(
            {
                "qualified_name": action_qn,
                "call_type": call_type,
                "handler_name": handler_name,
                "routine_id": routine_id,
                "routine_name": routine_name,
                "routine_owner_qn": str(row.get("routine_owner_qn") or ""),
                "extends_base_action": bool(row.get("extends_base_action")),
                "extension_action_count": int(row.get("extension_action_count") or 0),
            }
        )

    order = {"Main": 0, "Before": 1, "After": 2, "Override": 3}
    for event in by_event.values():
        event["actions"].sort(
            key=lambda action: (
                order.get(str(action.get("call_type") or ""), 10),
                str(action.get("handler_name") or action.get("routine_name") or ""),
            )
        )
    return sorted(by_event.values(), key=lambda event: str(event.get("name") or ""))


def _routine_payload(routine: Any) -> dict[str, Any] | None:
    if routine is None:
        return None
    props = dict(routine)
    return {
        "id": props.get("id") or "",
        "name": props.get("name") or "",
        "routine_type": props.get("routine_type") or "",
        "export": bool(props.get("export") or False),
        "signature": props.get("signature") or "",
        "directives": props.get("directives") or [],
        "doc_description": props.get("doc_description") or "",
        "doc_params_text": props.get("doc_params_text") or "",
        "doc_return_text": props.get("doc_return_text") or "",
        "area_path": props.get("area_path") or "",
        "file_path": props.get("file_path") or "",
        "line": int(props.get("line") or 0),
        "body": props.get("body") or "",
    }


def _routine_comment_lines(text: str) -> list[str]:
    return [f"// {line}".rstrip() for line in str(text or "").splitlines()]


def _routine_doc_comment(routine: dict[str, Any]) -> str:
    lines: list[str] = []
    description = str(routine.get("doc_description") or "").strip()
    params = str(routine.get("doc_params_text") or "").strip()
    return_text = str(routine.get("doc_return_text") or "").strip()
    if description:
        lines.extend(_routine_comment_lines(description))
    if params:
        if lines:
            lines.append("//")
        lines.append("// Параметры:")
        lines.extend(_routine_comment_lines(params))
    if return_text:
        if lines:
            lines.append("//")
        lines.append("// Возвращаемое значение:")
        lines.extend(_routine_comment_lines(return_text))
    return "\n".join(lines)


def _routine_code_text(routine: dict[str, Any]) -> str:
    parts: list[str] = []
    comment = _routine_doc_comment(routine)
    if comment:
        parts.append(comment)
    directives = [
        str(item).strip()
        for item in (routine.get("directives") or [])
        if str(item).strip()
    ]
    if directives:
        parts.append("\n".join(directives))
    body = str(routine.get("body") or "").strip()
    if body:
        parts.append(body)
    return "\n".join(parts).strip()


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _routine_code_layout(routine: dict[str, Any]) -> tuple[str, int]:
    parts: list[tuple[str, str]] = []
    comment = _routine_doc_comment(routine)
    if comment:
        parts.append(("comment", comment))
    directives = [
        str(item).strip()
        for item in (routine.get("directives") or [])
        if str(item).strip()
    ]
    if directives:
        parts.append(("directives", "\n".join(directives)))
    body = str(routine.get("body") or "").strip()
    if body:
        parts.append(("body", body))

    body_start_line = 1
    for kind, text in parts:
        if kind == "body":
            break
        body_start_line += _line_count(text)
    rendered = "\n".join(text for _, text in parts).strip()
    return rendered, body_start_line


def _module_code_from_routines(routines: list[dict[str, Any]]) -> str:
    bodies = []
    for routine in routines:
        body = _routine_code_text(routine)
        if body:
            bodies.append(body)
    return "\n\n".join(bodies)


def _module_routine_line_layout(routines: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    layout: dict[str, dict[str, int]] = {}
    current_line = 1
    for routine in routines:
        rendered, body_start_offset = _routine_code_layout(routine)
        if not rendered:
            continue
        routine_id = str(routine.get("id") or "")
        if routine_id:
            rendered_lines = _line_count(rendered)
            layout[routine_id] = {
                "routine_start_line": current_line,
                "body_start_line": current_line + body_start_offset - 1,
                "routine_end_line": current_line + rendered_lines - 1,
            }
        current_line += _line_count(rendered) + 1
    return layout


def _module_payload(row: dict[str, Any]) -> dict[str, Any]:
    props = dict(row.get("properties") or {})
    props.pop("module_type", None)
    routines = [
        item
        for item in (_routine_payload(routine) for routine in (row.get("routines") or []))
        if item
    ]
    routines.sort(key=lambda item: (
        str(item.get("file_path") or ""),
        int(item.get("line") or 0),
        str(item.get("name") or "").lower(),
    ))

    identity = {
        "id": row.get("id") or "",
        "name": row.get("name") or "",
        "module_type": row.get("module_type") or "",
        "path": row.get("path") or "",
        "owner_qn": row.get("owner_qn") or "",
        "owner_label": row.get("owner_label") or "",
        "config_name": row.get("config_name") or props.get("config_name") or "",
    }
    if not identity["name"]:
        identity["name"] = identity["module_type"] or "Модуль"

    return {
        "identity": identity,
        "badges": _extension_badges(props),
        "properties": _group_props(props),
        "routines": routines,
        "code": _module_code_from_routines(routines),
        "source": "routines",
    }


def get_module(
    module_id: str | None = None,
    owner_ref: str | None = None,
    module_type: str | None = None,
) -> dict[str, Any]:
    module_id = str(module_id or "").strip()
    owner_ref = str(owner_ref or "").strip()
    module_type = str(module_type or "").strip()
    params = {
        "project_name": settings.project_name,
        "module_id": module_id,
        "owner_ref": owner_ref,
        "module_type": module_type,
    }

    if module_id:
        rows = _run(
            """
            MATCH (mod:Module {id: $module_id, project_name: $project_name})
            OPTIONAL MATCH (owner)-[:HAS_MODULE]->(mod)
            OPTIONAL MATCH (mod)-[:DECLARES]->(r:Routine)
            WITH mod, owner, r
            ORDER BY coalesce(r.file_path, ''), coalesce(r.line, 0), coalesce(r.name, '')
            RETURN mod.id AS id,
                   coalesce(mod.name, mod.module_type, 'Модуль') AS name,
                   coalesce(mod.module_type, '') AS module_type,
                   coalesce(mod.path, '') AS path,
                   coalesce(mod.config_name, '') AS config_name,
                   coalesce(owner.qualified_name, mod.owner_qn, '') AS owner_qn,
                   CASE WHEN owner IS NULL THEN '' ELSE labels(owner)[0] END AS owner_label,
                   properties(mod) AS properties,
                   collect(r) AS routines
            LIMIT 1
            """,
            params,
        )
    elif owner_ref and module_type and module_type != "CommonModule":
        rows = _run(
            """
            MATCH (owner:""" + QN_LABEL_EXPR + """ {qualified_name: $owner_ref, project_name: $project_name})-[:HAS_MODULE]->(mod:Module)
            WHERE mod.module_type = $module_type
            OPTIONAL MATCH (mod)-[:DECLARES]->(r:Routine)
            WITH mod, owner, r
            ORDER BY coalesce(r.file_path, ''), coalesce(r.line, 0), coalesce(r.name, '')
            RETURN mod.id AS id,
                   coalesce(mod.name, mod.module_type, 'Модуль') AS name,
                   coalesce(mod.module_type, '') AS module_type,
                   coalesce(mod.path, '') AS path,
                   coalesce(mod.config_name, '') AS config_name,
                   owner.qualified_name AS owner_qn,
                   CASE WHEN owner IS NULL THEN '' ELSE labels(owner)[0] END AS owner_label,
                   properties(mod) AS properties,
                   collect(r) AS routines
            LIMIT 1
            """,
            params,
        )
    elif owner_ref:
        rows = _run(
            """
            MATCH (owner:MetadataObject {qualified_name: $owner_ref, project_name: $project_name})
            OPTIONAL MATCH (owner)-[:DECLARES]->(r:Routine)
            WITH owner, r
            ORDER BY coalesce(r.file_path, ''), coalesce(r.line, 0), coalesce(r.name, '')
            RETURN '' AS id,
                   coalesce(owner.name, owner.`Имя`, 'Общий модуль') AS name,
                   'CommonModule' AS module_type,
                   '' AS path,
                   coalesce(owner.config_name, '') AS config_name,
                   owner.qualified_name AS owner_qn,
                   CASE WHEN owner IS NULL THEN '' ELSE labels(owner)[0] END AS owner_label,
                   properties(owner) AS properties,
                   collect(r) AS routines
            LIMIT 1
            """,
            params,
        )
    else:
        raise ValueError("module_ref_required")

    if not rows:
        raise ValueError("module_not_found")
    return _module_payload(rows[0])


def get_module_code_units(
    module_id: str | None = None,
    owner_ref: str | None = None,
    module_type: str | None = None,
) -> dict[str, Any]:
    module = get_module(module_id, owner_ref, module_type)
    routines = list(module.get("routines") or [])
    layout = _module_routine_line_layout(routines)
    try:
        from graphdb.bsl_code_sqlite import get_bsl_code_sqlite
    except Exception as exc:
        return {
            "available": False,
            "reason": "sqlite_unavailable",
            "error": str(exc),
            "routines": [],
            "unit_count": 0,
            "split_routine_count": 0,
        }

    store = get_bsl_code_sqlite()
    epoch = store.current_epoch(settings.project_name)
    if epoch <= 0:
        return {
            "available": False,
            "reason": "not_indexed",
            "epoch": epoch,
            "routines": [],
            "unit_count": 0,
            "split_routine_count": 0,
        }

    result_routines: list[dict[str, Any]] = []
    total_units = 0
    max_lane = 0
    for routine in routines:
        routine_id = str(routine.get("id") or "")
        routine_layout = layout.get(routine_id)
        if not routine_id or not routine_layout:
            continue
        units = [
            unit for unit in store.all_units_by_parent(routine_id, epoch)
            if int(unit.part_total or 0) > 1
        ]
        if not units:
            continue
        units.sort(key=lambda unit: (
            int(unit.line_start or 0),
            int(unit.line_end or 0),
            int(unit.part_index or 0),
        ))
        lane_ends: list[int] = []
        payload_units: list[dict[str, Any]] = []
        for unit in units:
            line_start = max(1, int(unit.line_start or 1))
            line_end = max(line_start, int(unit.line_end or line_start))
            display_start = int(routine_layout["body_start_line"]) + line_start - 1
            display_end = int(routine_layout["body_start_line"]) + line_end - 1
            lane = 0
            while lane < len(lane_ends) and display_start <= lane_ends[lane]:
                lane += 1
            if lane == len(lane_ends):
                lane_ends.append(display_end)
            else:
                lane_ends[lane] = display_end
            max_lane = max(max_lane, lane)
            payload_units.append({
                "unit_id": unit.unit_id,
                "part_index": int(unit.part_index or 0),
                "part_total": int(unit.part_total or 0),
                "lane": lane,
                "line_start": line_start,
                "line_end": line_end,
                "display_line_start": display_start,
                "display_line_end": display_end,
                "char_start": int(unit.char_start or 0),
                "char_end": int(unit.char_end or 0),
                "unit_kind": unit.unit_kind,
            })
        if payload_units:
            total_units += len(payload_units)
            result_routines.append({
                "routine_id": routine_id,
                "name": routine.get("name") or "",
                "part_total": max(item["part_total"] for item in payload_units),
                "units": payload_units,
            })

    return {
        "available": True,
        "epoch": epoch,
        "routine_count": len(routines),
        "split_routine_count": len(result_routines),
        "unit_count": total_units,
        "max_lane": max_lane,
        "routines": result_routines,
    }


def get_object_summary_status(ref: str) -> dict[str, Any]:
    """Lightweight status payload for the manual summary button.

    Used by the polling endpoint — fetches only the fields needed to
    compute action_state, without touching attributes/relationships.
    Returns the same shape as `action_state` from `_object_summary`.
    """
    if not ref:
        raise ValueError("ref_required")
    rows = _run(
        """
        MATCH (m:MetadataObject {qualified_name: $ref, project_name: $project_name})
        RETURN m.qualified_name AS qualified_name,
               m.category_name AS category_name,
               m.config_name AS config_name,
               m.object_summary_path AS object_summary_path
        LIMIT 1
        """,
        {"project_name": settings.project_name, "ref": ref},
    )
    if not rows:
        raise ValueError("object_not_found")
    row = rows[0]
    props = {
        "category_name": row.get("category_name"),
        "config_name": row.get("config_name"),
        "object_summary_path": row.get("object_summary_path"),
    }
    has_summary = False
    path_value = props.get("object_summary_path")
    if path_value:
        summary_path = Path(str(path_value))
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("human_summary"), dict):
                has_summary = True
        except (OSError, json.JSONDecodeError, ValueError):
            has_summary = False
    return build_action_state(
        qualified_name=row["qualified_name"],
        props=props,
        has_summary=has_summary,
    )


def get_object(ref: str) -> dict[str, Any]:
    params = {"project_name": settings.project_name, "ref": ref}
    rows = _run(
        """
        MATCH (m:MetadataObject {qualified_name: $ref, project_name: $project_name})
        RETURN m.name AS name,
               m.qualified_name AS qualified_name,
               m.category_name AS category,
               m.config_name AS config_name,
               coalesce(m.`Синоним`, '') AS synonym,
               coalesce(m.`Комментарий`, '') AS comment,
               coalesce(m.`Пояснение`, '') AS explanation,
               properties(m) AS properties,
               size([(m)-[:HAS_ATTRIBUTE]->(a:Attribute) WHERE coalesce(a.`Стандартный`, false) <> true | 1]) AS attributes,
               size([(m)-[:HAS_ATTRIBUTE]->(a:Attribute) WHERE coalesce(a.`Стандартный`, false) = true | 1]) AS standard_attributes,
               size([(m)-[:HAS_TABULAR_PART]->(:TabularPart) | 1]) AS tabular_parts,
               size([(m)-[:HAS_RESOURCE]->(:Resource) | 1]) AS resources,
               size([(m)-[:HAS_DIMENSION]->(:Dimension) | 1]) AS dimensions,
               size([(m)-[:HAS_FORM]->(:Form) | 1]) AS forms,
               size([(m)-[:HAS_COMMAND]->(:Command) | 1]) AS commands,
               size([(m)-[:HAS_LAYOUT]->(:Layout) | 1]) AS layouts,
               size([(m)-[:HAS_GRAPH]->(:JournalGraph) | 1]) AS journal_graphs,
               size([(m)-[:HAS_ENUM_VALUE]->(:EnumValue) | 1]) AS enum_values,
               size([(m)-[:HAS_PREDEFINED]->(:PredefinedItem) | 1]) AS predefined,
               size([(m)-[:HAS_MODULE]->(:Module) | 1]) AS modules,
               size([(m)-[:DO_MOVEMENTS_IN]->(:MetadataObject) | 1]) AS movements
        LIMIT 1
        """,
        params,
    )
    if not rows:
        raise ValueError("object_not_found")
    obj = rows[0]
    props = obj.get("properties") or {}

    structure = _run(
        """
        MATCH (m:MetadataObject {qualified_name: $ref, project_name: $project_name})
        CALL {
          WITH m
          OPTIONAL MATCH (m)-[:HAS_ATTRIBUTE]->(n:Attribute)
          WHERE coalesce(n.`Стандартный`, false) <> true
          RETURN 'attributes' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_ATTRIBUTE]->(n:Attribute)
          WHERE coalesce(n.`Стандартный`, false) = true
          RETURN 'standard_attributes' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_TABULAR_PART]->(n:TabularPart)
          RETURN 'tabular_parts' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 '' AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_RESOURCE]->(n:Resource)
          RETURN 'resources' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_DIMENSION]->(n:Dimension)
          RETURN 'dimensions' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_FORM]->(n:Form)
          RETURN 'forms' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`ТипФормы`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_COMMAND]->(n:Command)
          RETURN 'commands' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Действие`, '') AS type, coalesce(n.`Заголовок`, n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_LAYOUT]->(n:Layout)
          RETURN 'layouts' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_GRAPH]->(n:JournalGraph)
          RETURN 'journal_graphs' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_ENUM_VALUE]->(n:EnumValue)
          RETURN 'enum_values' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 '' AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_PREDEFINED]->(n:PredefinedItem)
          RETURN 'predefined' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 '' AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_TABULAR_PART]->(tp:TabularPart)-[:HAS_ATTRIBUTE]->(n:Attribute)
          RETURN 'tabular_part_attributes' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 tp.qualified_name AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names
        }
        WITH section, name, qualified_name, type, synonym, parent_qualified_name, ownership, is_adopted,
             extension_adoptions, extension_names
        WHERE name IS NOT NULL
        ORDER BY section, name
        RETURN section, collect({
          name: name,
          qualified_name: qualified_name,
          type: type,
          synonym: synonym,
          parent_qualified_name: parent_qualified_name,
          ownership: ownership,
          is_adopted: is_adopted,
          extension_adoptions: extension_adoptions,
          extension_names: extension_names
        }) AS items
        ORDER BY section
        """,
        params,
    )

    modules = _run(
        """
        MATCH (m:MetadataObject {qualified_name: $ref, project_name: $project_name})
        CALL {
          WITH m
          OPTIONAL MATCH (m)-[:DECLARES]->(r:Routine)
          RETURN null AS module_id, 'CommonModule' AS module_type, '' AS module_name, m.qualified_name AS owner_qn, r,
                 '' AS ownership, false AS is_adopted, 0 AS extension_adoptions, [] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_MODULE]->(mod:Module)-[:DECLARES]->(r:Routine)
          RETURN mod.id AS module_id, coalesce(mod.module_type, '') AS module_type,
                 coalesce(mod.name, '') AS module_name, coalesce(mod.owner_qn, m.qualified_name) AS owner_qn, r,
                 coalesce(mod.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(mod)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(mod) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(mod) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_FORM]->(f:Form)-[:HAS_MODULE]->(mod:Module)-[:DECLARES]->(r:Routine)
          RETURN mod.id AS module_id, coalesce(mod.module_type, '') AS module_type,
                 coalesce(f.name, '') + ': ' + coalesce(mod.name, '') AS module_name, coalesce(mod.owner_qn, f.qualified_name) AS owner_qn, r,
                 coalesce(mod.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(mod)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(mod) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(mod) | extension.config_name] AS extension_names
          UNION ALL
          WITH m
          OPTIONAL MATCH (m)-[:HAS_COMMAND]->(cmd:Command)-[:HAS_MODULE]->(mod:Module)-[:DECLARES]->(r:Routine)
          RETURN mod.id AS module_id, coalesce(mod.module_type, '') AS module_type,
                 coalesce(cmd.name, '') + ': ' + coalesce(mod.name, '') AS module_name, coalesce(mod.owner_qn, cmd.qualified_name) AS owner_qn, r,
                 coalesce(mod.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(mod)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(mod) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(mod) | extension.config_name] AS extension_names
        }
        WITH module_id, module_type, module_name, owner_qn, ownership, is_adopted, extension_adoptions, extension_names,
             collect(DISTINCT r) AS routines
        WHERE size([r IN routines WHERE r IS NOT NULL]) > 0
        RETURN module_id, module_type, module_name, owner_qn,
               ownership, is_adopted, extension_adoptions, extension_names,
               size([r IN routines WHERE r IS NOT NULL]) AS routine_count,
               size([r IN routines WHERE r IS NOT NULL AND coalesce(r.export, false) = true]) AS exported_count
        ORDER BY module_type, module_name
        """,
        params,
    )

    return {
        "identity": {
            "name": obj["name"],
            "qualified_name": obj["qualified_name"],
            "category": obj["category"],
            "config_name": obj["config_name"],
            "synonym": obj.get("synonym") or "",
            "comment": obj.get("comment") or "",
            "explanation": obj.get("explanation") or "",
        },
        "badges": _extension_badges(props),
        "object_summary": _object_summary(props, qualified_name=obj["qualified_name"]),
        "counters": {
            key: obj.get(key, 0)
            for key in [
                "attributes",
                "standard_attributes",
                "tabular_parts",
                "resources",
                "dimensions",
                "forms",
                "commands",
                "layouts",
                "journal_graphs",
                "enum_values",
                "predefined",
                "modules",
                "movements",
            ]
        },
        "properties": _group_props(props),
        "structure": {
            row["section"]: row["items"]
            for row in structure
        },
        "modules": modules,
    }


def get_form_tree(ref: str) -> dict[str, Any]:
    params = {"project_name": settings.project_name, "ref": ref}
    rows = _run(
        """
        CALL {
          MATCH (root:Form {qualified_name: $ref, project_name: $project_name})
          OPTIONAL MATCH (owner:MetadataObject)-[:HAS_FORM]->(root)
          RETURN root, owner, 'form' AS root_kind
          UNION
          MATCH (root:MetadataObject {
            qualified_name: $ref,
            project_name: $project_name,
            category_name: 'ОбщиеФормы'
          })
          RETURN root, null AS owner, 'common_form' AS root_kind
        }
        RETURN root.name AS name,
               root.qualified_name AS qualified_name,
               coalesce(root.config_name, owner.config_name, '') AS config_name,
               coalesce(root.`ТипФормы`, root.console_search_type, '') AS type,
               coalesce(root.`Синоним`, root.`Заголовок`, '') AS synonym,
               root_kind
        LIMIT 1
        """,
        params,
    )
    if not rows:
        raise ValueError("form_not_found")
    form = rows[0]

    sections = _run(
        """
        CALL {
          MATCH (root:Form {qualified_name: $ref, project_name: $project_name})
          RETURN root
          UNION
          MATCH (root:MetadataObject {
            qualified_name: $ref,
            project_name: $project_name,
            category_name: 'ОбщиеФормы'
          })
          RETURN root
        }
        CALL {
          WITH root
          OPTIONAL MATCH (root)-[:HAS_FORM_ATTRIBUTE]->(n:FormAttribute)
          RETURN 'attributes' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, '') AS type, coalesce(n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names,
                 coalesce(n.`Порядок`, 0) AS order_index
          UNION ALL
          WITH root
          OPTIONAL MATCH (root)-[:HAS_COMMAND]->(n:Command)
          RETURN 'commands' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Действие`, '') AS type, coalesce(n.`Заголовок`, n.`Синоним`, '') AS synonym,
                 '' AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names,
                 coalesce(n.`Порядок`, 0) AS order_index
          UNION ALL
          WITH root
          MATCH (root)-[:HAS_CONTROL]->(root_control:FormControl)
          MATCH (root_control)-[:HAS_CHILD*0..]->(n:FormControl)
          OPTIONAL MATCH (p:FormControl)-[:HAS_CHILD]->(n)
          RETURN DISTINCT 'controls' AS section, n.name AS name, n.qualified_name AS qualified_name,
                 coalesce(n.`Тип`, n.`ТипКонтрола`, '') AS type, coalesce(n.`Синоним`, n.`Заголовок`, '') AS synonym,
                 coalesce(p.qualified_name, '') AS parent_qualified_name,
                 coalesce(n.`ПринадлежностьОбъекта`, '') AS ownership,
                 size([(n)-[:ADOPTED_FROM]->() | 1]) > 0 AS is_adopted,
                 size([(extension)-[:ADOPTED_FROM]->(n) | 1]) AS extension_adoptions,
                 [(extension)-[:ADOPTED_FROM]->(n) | extension.config_name] AS extension_names,
                 coalesce(n.`Порядок`, 0) AS order_index
        }
        WITH section, name, qualified_name, type, synonym, parent_qualified_name, ownership, is_adopted,
             extension_adoptions, extension_names, order_index
        WHERE name IS NOT NULL
        ORDER BY section, parent_qualified_name, order_index, name
        RETURN section, collect({
          name: name,
          qualified_name: qualified_name,
          type: type,
          synonym: synonym,
          parent_qualified_name: parent_qualified_name,
          ownership: ownership,
          is_adopted: is_adopted,
          extension_adoptions: extension_adoptions,
          extension_names: extension_names,
          order: order_index
        }) AS items
        ORDER BY section
        """,
        params,
    )

    structure = {row["section"]: row["items"] for row in sections}
    return {
        "identity": {
            "name": form.get("name") or "",
            "qualified_name": form.get("qualified_name") or "",
            "config_name": form.get("config_name") or "",
            "type": form.get("type") or "",
            "synonym": form.get("synonym") or "",
            "root_kind": form.get("root_kind") or "form",
        },
        "counters": {
            "attributes": len(structure.get("attributes") or []),
            "commands": len(structure.get("commands") or []),
            "controls": len(structure.get("controls") or []),
        },
        "structure": structure,
    }


def get_relationships(ref: str, limit: Any, offset: Any) -> dict[str, Any]:
    lim = _limit(limit, 50)
    off = _offset(offset)
    params = {
        "project_name": settings.project_name,
        "ref": ref,
        "limit": lim,
        "offset": off,
    }

    groups = []

    rel_queries = [
        (
            "uses",
            "Использует",
            """
            MATCH (m:MetadataObject {qualified_name: $ref, project_name: $project_name})
            CALL {
              WITH m
              MATCH (m)-[:HAS_ATTRIBUTE]->(elem)<-[:USED_IN]-(target:MetadataObject)
              RETURN target, 'Реквизиты.' + coalesce(elem.name, elem.`Имя`, '') AS via
              UNION
              WITH m
              MATCH (m)-[:HAS_TABULAR_PART]->(tp:TabularPart)-[:HAS_ATTRIBUTE]->(elem)<-[:USED_IN]-(target:MetadataObject)
              RETURN target,
                     'ТабличныеЧасти.' + coalesce(tp.name, tp.`Имя`, '') + '.Реквизиты.' + coalesce(elem.name, elem.`Имя`, '') AS via
              UNION
              WITH m
              MATCH (m)-[:HAS_RESOURCE]->(elem)<-[:USED_IN]-(target:MetadataObject)
              RETURN target, 'Ресурсы.' + coalesce(elem.name, elem.`Имя`, '') AS via
              UNION
              WITH m
              MATCH (m)-[:HAS_DIMENSION]->(elem)<-[:USED_IN]-(target:MetadataObject)
              RETURN target, 'Измерения.' + coalesce(elem.name, elem.`Имя`, '') AS via
            }
            WITH DISTINCT target, via
            RETURN target.name AS name, target.qualified_name AS qualified_name,
                   target.category_name AS category, target.config_name AS config_name,
                   via
            ORDER BY category, name
            SKIP $offset LIMIT $limit
            """,
        ),
        (
            "used_by",
            "Используется в",
            """
            MATCH (target:MetadataObject {qualified_name: $ref, project_name: $project_name})-[:USED_IN]->(elem)
            CALL {
              WITH elem
              OPTIONAL MATCH (owner:MetadataObject)-[:HAS_ATTRIBUTE]->(elem)
              RETURN owner, 'Реквизиты.' + coalesce(elem.name, elem.`Имя`, '') AS via
              UNION
              WITH elem
              OPTIONAL MATCH (owner:MetadataObject)-[:HAS_TABULAR_PART]->(tp:TabularPart)-[:HAS_ATTRIBUTE]->(elem)
              RETURN owner,
                     'ТабличныеЧасти.' + coalesce(tp.name, tp.`Имя`, '') + '.Реквизиты.' + coalesce(elem.name, elem.`Имя`, '') AS via
              UNION
              WITH elem
              OPTIONAL MATCH (owner:MetadataObject)-[:HAS_RESOURCE]->(elem)
              RETURN owner, 'Ресурсы.' + coalesce(elem.name, elem.`Имя`, '') AS via
              UNION
              WITH elem
              OPTIONAL MATCH (owner:MetadataObject)-[:HAS_DIMENSION]->(elem)
              RETURN owner, 'Измерения.' + coalesce(elem.name, elem.`Имя`, '') AS via
            }
            WITH DISTINCT owner, via
            WHERE owner IS NOT NULL AND owner.project_name = $project_name
            RETURN owner.name AS name, owner.qualified_name AS qualified_name,
                   owner.category_name AS category, owner.config_name AS config_name,
                   via
            ORDER BY category, name
            SKIP $offset LIMIT $limit
            """,
        ),
        (
            "movements",
            "Делает движения в",
            """
            MATCH (m:MetadataObject {qualified_name: $ref, project_name: $project_name})-[rel:DO_MOVEMENTS_IN]->(target:MetadataObject)
            RETURN target.name AS name, target.qualified_name AS qualified_name,
                   target.category_name AS category, target.config_name AS config_name,
                   '' AS via,
                   coalesce(rel.`ЗаписьДвиженийПриПроведении`, '') AS write_mode,
                   coalesce(rel.`УдалениеДвижений`, '') AS delete_mode
            ORDER BY category, name
            SKIP $offset LIMIT $limit
            """,
        ),
        (
            "moved_by",
            "Документы регистраторы",
            """
            MATCH (owner:MetadataObject {project_name: $project_name})-[rel:DO_MOVEMENTS_IN]->(:MetadataObject {qualified_name: $ref})
            RETURN owner.name AS name, owner.qualified_name AS qualified_name,
                   owner.category_name AS category, owner.config_name AS config_name,
                   '' AS via,
                   coalesce(rel.`ЗаписьДвиженийПриПроведении`, '') AS write_mode,
                   coalesce(rel.`УдалениеДвижений`, '') AS delete_mode
            ORDER BY category, name
            SKIP $offset LIMIT $limit
            """,
        ),
        (
            "role_grants",
            "Права роли",
            """
            MATCH (role:MetadataObject {qualified_name: $ref, project_name: $project_name, category_name: 'Роли'})
                  -[rel:GRANTS_ACCESS_TO]->(target)
            WITH role, rel, target, properties(rel) AS props
            WITH target, rel,
                 [right IN coalesce(rel.rights_present_en, []) | {
                   name: coalesce(props[right + '_ru'], right),
                   has_condition: coalesce(props[right + '_has_condition'], false),
                   condition: coalesce(props[right + '_condition'], '')
                 }] AS rights_raw
            WITH target, rel,
                 reduce(rights = [], right IN rights_raw |
                   CASE
                     WHEN right.name = '' OR any(existing IN rights WHERE existing.name = right.name AND existing.condition = right.condition)
                     THEN rights
                     ELSE rights + right
                   END
                 ) AS rights
            RETURN coalesce(target.name, target.`Имя`, target.qualified_name) AS name,
                   target.qualified_name AS qualified_name,
                   coalesce(target.category_name, labels(target)[0]) AS category,
                   coalesce(target.config_name, '') AS config_name,
                   CASE
                     WHEN size(rights) > 0 THEN [right IN rights | right.name]
                     ELSE [coalesce(rel.right_ru, rel.right, '')]
                   END AS via,
                   rights
            ORDER BY category, name
            SKIP $offset LIMIT $limit
            """,
        ),
        (
            "access",
            "Права",
            """
            MATCH (role:MetadataObject {project_name: $project_name, category_name: 'Роли'})-[rel:GRANTS_ACCESS_TO]->(target {qualified_name: $ref})
            WITH role, rel, properties(rel) AS props
            WITH role, rel,
                 [right IN coalesce(rel.rights_present_en, []) | {
                   name: coalesce(props[right + '_ru'], right),
                   has_condition: coalesce(props[right + '_has_condition'], false),
                   condition: coalesce(props[right + '_condition'], '')
                 }] AS rights_raw
            WITH role, rel,
                 reduce(rights = [], right IN rights_raw |
                   CASE
                     WHEN right.name = '' OR any(existing IN rights WHERE existing.name = right.name AND existing.condition = right.condition)
                     THEN rights
                     ELSE rights + right
                   END
                 ) AS rights
            RETURN role.name AS name, role.qualified_name AS qualified_name,
                   role.category_name AS category, role.config_name AS config_name,
                   CASE
                     WHEN size(rights) > 0 THEN [right IN rights | right.name]
                     ELSE [coalesce(rel.right_ru, rel.right, '')]
                   END AS via,
                   rights
            ORDER BY name
            SKIP $offset LIMIT $limit
            """,
        ),
        (
            "subscriptions",
            "Подписки на события",
            """
            MATCH (:MetadataObject {qualified_name: $ref, project_name: $project_name})-[:HAS_EVENT_SUBSCRIPTION]->(target:MetadataObject)
            RETURN target.name AS name, target.qualified_name AS qualified_name,
                   target.category_name AS category, target.config_name AS config_name,
                   coalesce(target.`Событие`, '') AS via
            ORDER BY name
            SKIP $offset LIMIT $limit
            """,
        ),
    ]

    for key, title, query in rel_queries:
        rows = _run(query, params)
        groups.append({
            "key": key,
            "title": title,
            "limit": lim,
            "offset": off,
            "items": rows,
        })

    extension_items = _get_extension_relationship_items(ref)
    if extension_items:
        groups.append({
            "key": "extensions",
            "title": "Расширения",
            "limit": lim,
            "offset": off,
            "items": extension_items,
        })

    return {
        "qualified_name": ref,
        "groups": groups,
    }


def _get_extension_relationship_items(ref: str) -> list[dict[str, Any]]:
    params = {
        "project_name": settings.project_name,
        "ref": ref,
    }
    extension_rows = _run(
        """
        MATCH (ext:MetadataObject)-[:ADOPTED_FROM]->(base:MetadataObject {qualified_name: $ref, project_name: $project_name})
        RETURN ext.config_name AS extension_config,
               ext.qualified_name AS extension_ref,
               coalesce(ext.name, ext.`Имя`, '') AS name,
               coalesce(ext.category_name, '') AS category,
               coalesce(ext.`ПринадлежностьОбъекта`, '') AS ownership,
               coalesce(ext.modified_properties, []) AS modified,
               coalesce(ext.controlled_properties, []) AS controlled,
               [prop IN coalesce(ext.modified_properties, []) |
                 {
                   property: prop,
                   base: CASE WHEN prop IN keys(base) THEN base[prop] ELSE NULL END,
                   extension: CASE WHEN prop IN keys(ext) THEN ext[prop] ELSE NULL END,
                   has_base: prop IN keys(base),
                   has_extension: prop IN keys(ext)
                 }
               ] AS modified_values,
               [prop IN coalesce(ext.controlled_properties, []) |
                 {
                   property: prop,
                   base: CASE WHEN prop IN keys(base) THEN base[prop] ELSE NULL END,
                   extension: CASE WHEN prop IN keys(ext) THEN ext[prop] ELSE NULL END,
                   has_base: prop IN keys(base),
                   has_extension: prop IN keys(ext)
                 }
               ] AS controlled_values
        ORDER BY extension_config
        """,
        params,
    )
    if not extension_rows:
        return []

    child_rows = _run(
        """
        MATCH (ext:MetadataObject)-[:ADOPTED_FROM]->(:MetadataObject {qualified_name: $ref, project_name: $project_name})
        CALL (ext) {
          OPTIONAL MATCH (ext)-[:HAS_ATTRIBUTE]->(n:Attribute)
          WHERE coalesce(n.`Стандартный`, false) <> true
          RETURN 'Реквизиты' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_ATTRIBUTE]->(n:Attribute)
          WHERE coalesce(n.`Стандартный`, false) = true
          RETURN 'Стандартные реквизиты' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_TABULAR_PART]->(n:TabularPart)
          RETURN 'Табличные части' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_TABULAR_PART]->(tp:TabularPart)-[:HAS_ATTRIBUTE]->(n:Attribute)
          RETURN 'Реквизиты табличных частей' AS section,
                 coalesce(tp.name, tp.`Имя`, '') + '.Реквизиты.' + coalesce(n.name, n.`Имя`, '') AS path,
                 n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_RESOURCE]->(n:Resource)
          RETURN 'Ресурсы' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_DIMENSION]->(n:Dimension)
          RETURN 'Измерения' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_FORM]->(n:Form)
          RETURN 'Формы' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_COMMAND]->(n:Command)
          RETURN 'Команды' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_LAYOUT]->(n:Layout)
          RETURN 'Макеты' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_GRAPH]->(n:JournalGraph)
          RETURN 'Графы' AS section, coalesce(n.name, n.`Имя`, '') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_MODULE]->(n:Module)
          RETURN 'Модули' AS section, coalesce(n.name, n.module_type, 'Модуль') AS path, n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_FORM]->(form:Form)-[:HAS_CONTROL]->(root:FormControl)-[:HAS_CHILD*0..]->(n:FormControl)
          WHERE coalesce(n.ext_source, '') IN ['own', 'adopted_modified']
             OR size(coalesce(n.modified_properties, [])) > 0
             OR size(coalesce(n.controlled_properties, [])) > 0
             OR coalesce(n.`ПринадлежностьОбъекта`, '') = 'Собственный'
          RETURN 'Элементы форм' AS section,
                 coalesce(form.name, form.`Имя`, '') + '.Элементы.' + coalesce(n.name_path, n.name, n.`Имя`, '') AS path,
                 n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_FORM]->(form:Form)-[:HAS_FORM_ATTRIBUTE]->(n:FormAttribute)
          WHERE coalesce(n.ext_source, '') IN ['own', 'adopted_modified']
             OR size(coalesce(n.modified_properties, [])) > 0
             OR size(coalesce(n.controlled_properties, [])) > 0
             OR coalesce(n.`ПринадлежностьОбъекта`, '') = 'Собственный'
          RETURN 'Реквизиты форм' AS section,
                 coalesce(form.name, form.`Имя`, '') + '.Реквизиты.' + coalesce(n.name, n.`Имя`, '') AS path,
                 n AS node
          UNION ALL
          OPTIONAL MATCH (ext)-[:HAS_FORM]->(form:Form)-[:HAS_COMMAND]->(n:Command)
          WHERE coalesce(n.ext_source, '') IN ['own', 'adopted_modified']
             OR size(coalesce(n.modified_properties, [])) > 0
             OR size(coalesce(n.controlled_properties, [])) > 0
             OR coalesce(n.`ПринадлежностьОбъекта`, '') = 'Собственный'
          RETURN 'Команды форм' AS section,
                 coalesce(form.name, form.`Имя`, '') + '.Команды.' + coalesce(n.name, n.`Имя`, '') AS path,
                 n AS node
        }
        WITH ext, section, path, node
        WHERE node IS NOT NULL
        OPTIONAL MATCH (node)-[:ADOPTED_FROM]->(base_node)
        WITH ext, section, path, node,
             base_node,
             CASE WHEN base_node IS NULL THEN [] ELSE keys(base_node) END AS base_keys,
             keys(node) AS node_keys,
             coalesce(node.`ПринадлежностьОбъекта`, '') AS ownership,
             coalesce(node.modified_properties, []) AS modified,
             coalesce(node.controlled_properties, []) AS controlled,
             coalesce(node.ext_source, '') AS ext_source,
             size([(node)-[:ADOPTED_FROM]->() | 1]) > 0 AS adopted
        WHERE ownership <> ''
           OR adopted
           OR ext_source IN ['own', 'adopted_modified']
           OR size(modified) > 0
           OR size(controlled) > 0
        RETURN ext.config_name AS extension_config,
               section,
               path,
               labels(node)[0] AS label,
               node.qualified_name AS qualified_name,
               coalesce(node.name, node.`Имя`, path) AS name,
               ownership,
               adopted,
               ext_source,
               modified,
               controlled,
               [prop IN modified |
                 {
                   property: prop,
                   base: CASE WHEN prop IN base_keys THEN base_node[prop] ELSE NULL END,
                   extension: CASE WHEN prop IN node_keys THEN node[prop] ELSE NULL END,
                   has_base: prop IN base_keys,
                   has_extension: prop IN node_keys
                 }
               ] AS modified_values,
               [prop IN controlled |
                 {
                   property: prop,
                   base: CASE WHEN prop IN base_keys THEN base_node[prop] ELSE NULL END,
                   extension: CASE WHEN prop IN node_keys THEN node[prop] ELSE NULL END,
                   has_base: prop IN base_keys,
                   has_extension: prop IN node_keys
                 }
               ] AS controlled_values
        ORDER BY extension_config, section, path
        LIMIT 600
        """,
        params,
    )

    by_extension: dict[str, dict[str, Any]] = {}
    for row in extension_rows:
        ext_config = row.get("extension_config") or ""
        by_extension[ext_config] = {
            "extension_config": ext_config,
            "extension_ref": row.get("extension_ref") or "",
            "name": row.get("name") or ext_config,
            "category": row.get("category") or "",
            "ownership": row.get("ownership") or "",
            "object": {
                "modified": row.get("modified") or [],
                "controlled": row.get("controlled") or [],
                "modified_values": row.get("modified_values") or [],
                "controlled_values": row.get("controlled_values") or [],
            },
            "sections": [],
        }

    section_maps: dict[str, dict[str, list[dict[str, Any]]]] = {
        ext: {} for ext in by_extension
    }
    for row in child_rows:
        ext_config = row.get("extension_config") or ""
        if ext_config not in by_extension:
            continue
        section = row.get("section") or "Элементы"
        item = {
            "path": row.get("path") or row.get("name") or "",
            "name": row.get("name") or "",
            "qualified_name": row.get("qualified_name") or "",
            "label": row.get("label") or "",
            "ownership": row.get("ownership") or "",
            "is_adopted": bool(row.get("adopted")),
            "ext_source": row.get("ext_source") or "",
            "modified": row.get("modified") or [],
            "controlled": row.get("controlled") or [],
            "modified_values": row.get("modified_values") or [],
            "controlled_values": row.get("controlled_values") or [],
        }
        _suppress_default_form_attribute_title_change(item)
        section_maps.setdefault(ext_config, {}).setdefault(section, []).append(item)

    for ext_config, section_map in section_maps.items():
        by_extension[ext_config]["sections"] = [
            {"title": title, "items": items}
            for title, items in section_map.items()
            if items
        ]

    return list(by_extension.values())


def _suppress_default_form_attribute_title_change(item: dict[str, Any]) -> None:
    """Hide noisy form item title diffs when extension only materializes default title."""
    if item.get("label") not in {"FormAttribute", "FormControl"}:
        return
    modified = item.get("modified") or []
    if "Заголовок" not in modified:
        return

    diff_by_prop = {
        diff.get("property"): diff
        for diff in item.get("modified_values") or []
        if isinstance(diff, dict) and diff.get("property")
    }
    diff = diff_by_prop.get("Заголовок")
    if not diff:
        return

    name = item.get("name") or item.get("path") or ""
    base_effective = (
        _stringify_diff_value(diff.get("base"))
        if diff.get("has_base")
        else _default_form_attribute_title(name)
    )
    ext_effective = (
        _stringify_diff_value(diff.get("extension"))
        if diff.get("has_extension")
        else _default_form_attribute_title(name)
    )

    if not base_effective or base_effective != ext_effective:
        return

    item["modified"] = [prop for prop in modified if prop != "Заголовок"]
    item["modified_values"] = [
        diff_value
        for diff_value in (item.get("modified_values") or [])
        if not (isinstance(diff_value, dict) and diff_value.get("property") == "Заголовок")
    ]


def _default_form_attribute_title(name: str) -> str:
    """Approximate 1C default title derived from a FormAttribute name."""
    value = str(name or "").strip()
    if not value:
        return ""
    # Split PascalCase while preserving multi-letter acronyms like НДФЛ.
    spaced = re.sub(r"(?<=[а-яёa-z0-9])(?=[А-ЯЁA-Z])", " ", value)
    spaced = re.sub(r"(?<=[А-ЯЁA-Z])(?=[А-ЯЁA-Z][а-яёa-z])", " ", spaced)
    tokens = spaced.split()
    if not tokens:
        return value
    result = [tokens[0]]
    for token in tokens[1:]:
        if len(token) > 1 and token.upper() == token:
            result.append(token)
        else:
            result.append(token[:1].lower() + token[1:])
    return " ".join(result)


def _stringify_diff_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify_diff_value(item) for item in value)
    return str(value)
