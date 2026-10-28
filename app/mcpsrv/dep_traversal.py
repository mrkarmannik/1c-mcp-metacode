"""
BFS-based dependency path traversal for find_dependency_paths MCP tool.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from config import settings
from .queries import _run_query
from .resolvers import (
    QN_LABEL_EXPR,
    resolve_object_ref,
    resolve_element_ref,
    resolve_form_owner_ref,
    resolve_form_event_ref,
    resolve_control_ref,
    _md_is_form_path,
    _MD_SEC_CONTROL,
    _MD_SEC_EVENT,
    _MD_SEC_FORM_MARKER,
)

logger = logging.getLogger(__name__)

# Full-QN start labels allow-list (chosen by membership, not labels(n)[0], so Neo4j label
# order does not affect the result).
_START_QN_LABELS: Tuple[str, ...] = (
    "MetadataObject", "Attribute", "Resource", "Dimension", "AccountingFlag",
    "DimensionAccountingFlag", "TabularPart", "Form", "FormControl", "FormAttribute",
    "FormEvent", "FormEventAction", "Command", "UrlMethod", "Routine",
)

_ELEMENT_LABELS: frozenset[str] = frozenset({
    "Attribute", "Resource", "Dimension", "AccountingFlag",
    "DimensionAccountingFlag", "FormAttribute", "TabularPart",
})

_BSL_RELS: frozenset[str] = frozenset({"CALLS", "HAS_HANDLER", "USES_HANDLER"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraversalStep:
    from_ref: str
    from_label: str
    to_ref: str
    to_label: str
    to_name: str
    relationship_type: str
    owner_step: bool
    to_owner_ref: Optional[str] = None  # для Routine-хопов: callee.owner_qn


@dataclass
class DependencyPath:
    depth: int
    start_ref: str
    start_label: str
    end_ref: str
    end_label: str
    end_name: str
    end_owner_ref: Optional[str]
    relationship_chain: List[str]
    steps: List[TraversalStep]
    path_display: str


# ---------------------------------------------------------------------------
# Start node resolution
# ---------------------------------------------------------------------------

def _lookup_qn_start(
    loader: Any, ref: str, project_name: str, config_name: Optional[str]
) -> Optional[Tuple[str, str]]:
    """Return (qualified_name, label) when ref is an existing node QN with a supported start
    label (chosen by _START_QN_LABELS membership, not labels(n)[0], so Neo4j label order does
    not matter). Returns None when no node with this qualified_name exists. Raises on a
    config-scope violation or an existing node whose label is not a valid dependency start."""
    rows = loader.execute_query_readonly(
        f"MATCH (n:{QN_LABEL_EXPR} {{qualified_name: $qn}}) RETURN labels(n) AS labels LIMIT 1", {"qn": ref}
    ) or []
    if not rows:
        return None
    if config_name and not ref.startswith(f"{project_name}/{config_name}/"):
        raise ValueError(f"Node '{ref}' does not belong to config '{config_name}'.")
    labels = rows[0].get("labels") or []
    for lbl in _START_QN_LABELS:
        if lbl in labels:
            return (ref, lbl)
    raise ValueError(
        f"qualified_name {ref!r} has labels {labels!r}, not supported as a dependency start."
    )


def resolve_start_node(
    loader: Any,
    start_ref: str,
    project_name: str,
    config_name: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve start_ref to (ref, label).

    Order: Routine id → absolute QN (allow-list, hard) → section-style form-control →
    form-event → form → soft relative QN heuristic → element/object fallback.
    Form/control/event refs are parsed like get_metadata_details' _md_parse_section_ref
    (re.split on '.'/'/', case-insensitive section markers). Raises ValueError if nothing resolves.
    """
    ref = (start_ref or "").strip()
    if not ref:
        raise ValueError("start_ref is required.")

    # 1. Routine by id: no '/' and not 'Category.Name' pattern
    if "/" not in ref and "." not in ref:
        params: Dict[str, Any] = {"ref": ref, "pn": project_name}
        cypher_r = "MATCH (r:Routine {id: $ref, project_name: $pn}) RETURN r.id AS ref LIMIT 1"
        if config_name:
            params["cn"] = config_name
            cypher_r = "MATCH (r:Routine {id: $ref, project_name: $pn, config_name: $cn}) RETURN r.id AS ref LIMIT 1"
        rows = loader.execute_query_readonly(cypher_r, params) or []
        if rows and rows[0].get("ref"):
            return (rows[0]["ref"], "Routine")

    segs = re.split(r"[./]", ref)
    low = [s.lower() for s in segs]

    # 2. Absolute QN (hard): full qualified_name always starts with project_name/.
    #    Uses English path segments (Form/Control/Event), so it never collides with the
    #    Russian section markers below; a miss here is a clean dependency-start error.
    if ref.startswith(project_name + "/"):
        found = _lookup_qn_start(loader, ref, project_name, config_name)
        if found:
            return found
        raise ValueError(f"qualified_name {ref!r} was not found as dependency start.")

    # 3. Section-style form-control: <form path>.<Control>.<Имя>
    if len(segs) >= 4 and low[-2] in _MD_SEC_CONTROL and _md_is_form_path(low[:-2]):
        form_path = ".".join(segs[:-2])
        ctrl_qn = resolve_control_ref(loader, form_path, segs[-1], project_name, config_name)
        if ctrl_qn is None:
            raise ValueError(f"Form control {segs[-1]!r} not found in form {form_path!r}.")
        return (ctrl_qn, "FormControl")

    # 4. Section-style form-event: <form path>.<Event>.<Имя> (form-level or control-level).
    #    resolve_form_event_ref parses both levels and both separators.
    if len(segs) >= 4 and low[-2] in _MD_SEC_EVENT:
        event_qn = resolve_form_event_ref(loader, ref, project_name, config_name)
        return (event_qn, "FormEvent")

    # 5. Human-readable form: needs an explicit форма/формы marker or a leading ОбщиеФормы.
    if low[:1] == ["общиеформы"] or any(t in _MD_SEC_FORM_MARKER for t in low):
        form_qn = resolve_form_owner_ref(loader, ref, project_name, config_name)
        # common form resolves to a MetadataObject; object form to a Form node
        label = "MetadataObject" if "/ОбщиеФормы/" in form_qn else "Form"
        return (form_qn, label)

    # 6. Soft relative QN heuristic (rare QN-like refs without form markers): try QN lookup,
    #    fall through on miss (do NOT hard-error — that would swallow section-style slash refs).
    if ref.count("/") >= 4:
        found = _lookup_qn_start(loader, ref, project_name, config_name)
        if found:
            return found

    # 7. Element short ref, then MetadataObject short ref
    sep_count = ref.count(".") + ref.count("/")
    if 2 <= sep_count <= 3:
        resolved = resolve_element_ref(loader, ref, project_name, config_name)
        if resolved:
            return resolved

    obj = resolve_object_ref(loader, ref, project_name, config_name)
    return (obj["qualified_name"], "MetadataObject")


# ---------------------------------------------------------------------------
# Owner bridge helpers
# ---------------------------------------------------------------------------

def _owner_qn_from_element_qn(elem_qn: str) -> str:
    """
    Extract MetadataObject QN from element QN: take first 4 path parts.
    project/config/Category/Object/element_type/name → project/config/Category/Object
    Also handles FormAttribute nested in Form (still returns top-level MetadataObject).
    """
    return "/".join(elem_qn.split("/")[:4])


def _make_owner_bridge(
    hop: TraversalStep,
    loader: Any = None,
    owner_label_cache: Optional[Dict[str, str]] = None,
) -> Optional[TraversalStep]:
    """
    Returns an OWNER_BRIDGE step to continue BFS at the owner node after a real hop.
    Returns None if the hop already ends at a MetadataObject or no bridge is possible.
    """
    if hop.to_label in _ELEMENT_LABELS:
        owner_qn = _owner_qn_from_element_qn(hop.to_ref)
        if owner_qn == hop.from_ref:
            return None
        return TraversalStep(
            from_ref=hop.to_ref, from_label=hop.to_label,
            to_ref=owner_qn, to_label="MetadataObject",
            to_name="",
            relationship_type="OWNER_BRIDGE", owner_step=True,
        )
    if hop.to_label == "Routine" and hop.to_owner_ref:
        # Resolve actual owner label — can be MetadataObject, Command, Form, or Configuration
        owner_label = "MetadataObject"
        cache = owner_label_cache if owner_label_cache is not None else {}
        if hop.to_owner_ref in cache:
            owner_label = cache[hop.to_owner_ref]
        elif loader is not None:
            rows = loader.execute_query_readonly(
                "MATCH (n:MetadataObject|Command|Form|Configuration {qualified_name: $qn})"
                " RETURN labels(n)[0] AS label LIMIT 1",
                {"qn": hop.to_owner_ref},
            ) or []
            if rows and rows[0].get("label"):
                owner_label = rows[0]["label"]
            cache[hop.to_owner_ref] = owner_label
        return TraversalStep(
            from_ref=hop.to_ref, from_label="Routine",
            to_ref=hop.to_owner_ref, to_label=owner_label,
            to_name="",
            relationship_type="OWNER_BRIDGE", owner_step=True,
        )
    return None


# ---------------------------------------------------------------------------
# Cypher helpers — one hop per relationship family
# ---------------------------------------------------------------------------

def _cfg_and(alias: str, config_name: Optional[str]) -> str:
    return f" AND {alias}.config_name = $config_name" if config_name else ""


def _expand_used_in_downstream(
    loader: Any, obj_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    # FormAttribute has no project_name in the graph — filter it by QN prefix.
    # Other element labels (Attribute, Resource, etc.) have project_name and config_name.
    cfg_prop = _cfg_and("elem", config_name)
    qn_prefix = (project_name + "/" + config_name + "/") if config_name else (project_name + "/")
    rows = _run_query(loader, f"""
MATCH (src:MetadataObject {{qualified_name: $qn, project_name: $project_name}})
-[:USED_IN]->(elem)
WHERE (
  (elem:Attribute OR elem:Resource OR elem:Dimension
   OR elem:AccountingFlag OR elem:DimensionAccountingFlag)
  AND elem.project_name = $project_name{cfg_prop}
) OR (
  elem:FormAttribute
  AND elem.qualified_name STARTS WITH $qn_prefix
)
RETURN labels(elem)[0] AS label, elem.qualified_name AS ref, elem.name AS name
""".strip(), {"qn": obj_qn, "config_name": config_name, "qn_prefix": qn_prefix}, project_name)
    return [
        TraversalStep(
            from_ref=obj_qn, from_label="MetadataObject",
            to_ref=r["ref"], to_label=r["label"],
            to_name=r.get("name") or "",
            relationship_type="USED_IN", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_used_in_upstream(
    loader: Any, obj_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg = _cfg_and("typeObj", config_name)
    rows = _run_query(loader, f"""
CALL {{
  MATCH (src:MetadataObject {{qualified_name: $qn, project_name: $project_name}})
        -[:HAS_ATTRIBUTE|HAS_RESOURCE|HAS_DIMENSION
           |HAS_ACCOUNTING_FLAG|HAS_DIMENSION_ACCOUNTING_FLAG]->(elem)
  MATCH (typeObj:MetadataObject)-[:USED_IN]->(elem)
  WHERE typeObj.project_name = $project_name{cfg}
  RETURN DISTINCT typeObj.qualified_name AS ref, typeObj.name AS name
  UNION
  MATCH (src:MetadataObject {{qualified_name: $qn, project_name: $project_name}})
        -[:HAS_TABULAR_PART]->(tp:TabularPart)-[:HAS_ATTRIBUTE]->(elem:Attribute)
  MATCH (typeObj:MetadataObject)-[:USED_IN]->(elem)
  WHERE typeObj.project_name = $project_name{cfg}
  RETURN DISTINCT typeObj.qualified_name AS ref, typeObj.name AS name
}}
RETURN ref, name
""".strip(), {"qn": obj_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=obj_qn, from_label="MetadataObject",
            to_ref=r["ref"], to_label="MetadataObject",
            to_name=r.get("name") or "",
            relationship_type="USED_IN", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_do_movements_downstream(
    loader: Any, obj_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg = _cfg_and("reg", config_name)
    rows = _run_query(loader, f"""
MATCH (doc:MetadataObject {{qualified_name: $qn, project_name: $project_name}})
      -[:DO_MOVEMENTS_IN]->(reg:MetadataObject)
WHERE reg.project_name = $project_name{cfg}
RETURN reg.qualified_name AS ref, reg.name AS name
""".strip(), {"qn": obj_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=obj_qn, from_label="MetadataObject",
            to_ref=r["ref"], to_label="MetadataObject",
            to_name=r.get("name") or "",
            relationship_type="DO_MOVEMENTS_IN", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_do_movements_upstream(
    loader: Any, obj_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg = _cfg_and("doc", config_name)
    rows = _run_query(loader, f"""
MATCH (doc:MetadataObject)-[:DO_MOVEMENTS_IN]->
      (reg:MetadataObject {{qualified_name: $qn, project_name: $project_name}})
WHERE doc.project_name = $project_name{cfg}
RETURN doc.qualified_name AS ref, doc.name AS name
""".strip(), {"qn": obj_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=obj_qn, from_label="MetadataObject",
            to_ref=r["ref"], to_label="MetadataObject",
            to_name=r.get("name") or "",
            relationship_type="DO_MOVEMENTS_IN", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _calls_cfg_filter(config_name: Optional[str], src_alias: str, dst_alias: str) -> str:
    if not config_name:
        return ""
    return f" AND {src_alias}.config_name = $config_name AND {dst_alias}.config_name = $config_name"


def _expand_calls_from_routine(
    loader: Any, routine_id: str, direction: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    results: List[TraversalStep] = []
    params = {"routine_id": routine_id, "config_name": config_name}

    if direction in ("downstream", "both"):
        cfg = _calls_cfg_filter(config_name, "src", "callee")
        rows = _run_query(loader, f"""
MATCH (src:Routine {{id: $routine_id, project_name: $project_name}})-[:CALLS]->(callee:Routine)
WHERE callee.project_name = $project_name{cfg}
RETURN callee.id AS ref, callee.name AS name, callee.owner_qn AS owner_qn
""".strip(), params, project_name)
        results += [
            TraversalStep(
                from_ref=routine_id, from_label="Routine",
                to_ref=r["ref"], to_label="Routine",
                to_name=r.get("name") or "",
                relationship_type="CALLS", owner_step=False,
                to_owner_ref=r.get("owner_qn"),
            )
            for r in rows if r.get("ref")
        ]

    if direction in ("upstream", "both"):
        cfg = _calls_cfg_filter(config_name, "caller", "dst")
        rows = _run_query(loader, f"""
MATCH (caller:Routine)-[:CALLS]->(dst:Routine {{id: $routine_id, project_name: $project_name}})
WHERE caller.project_name = $project_name{cfg}
RETURN caller.id AS ref, caller.name AS name, caller.owner_qn AS owner_qn
""".strip(), params, project_name)
        results += [
            TraversalStep(
                from_ref=routine_id, from_label="Routine",
                to_ref=r["ref"], to_label="Routine",
                to_name=r.get("name") or "",
                relationship_type="CALLS", owner_step=False,
                to_owner_ref=r.get("owner_qn"),
            )
            for r in rows if r.get("ref")
        ]

    return results


def _expand_calls_from_owner(
    loader: Any, owner_qn: str, owner_label: str,
    project_name: str, config_name: Optional[str],
) -> List[TraversalStep]:
    """
    Bridge owner → each of its declared routines (OWNER_BRIDGE, owner_step=True).
    The real CALLS hop happens in the next BFS iteration from each routine via
    _expand_calls_from_routine, which records the precise caller/callee pair.
    Direction is not needed here — it is applied at Routine level.
    """
    cfg_r = _cfg_and("r", config_name)
    rows = _run_query(loader, f"""
MATCH (src:{QN_LABEL_EXPR} {{qualified_name: $qn}})
      -[:HAS_MODULE|DECLARES]->(mod_or_r)
WITH CASE WHEN mod_or_r:Module THEN mod_or_r ELSE null END AS mod,
     CASE WHEN mod_or_r:Routine THEN mod_or_r ELSE null END AS direct_r
OPTIONAL MATCH (mod)-[:DECLARES]->(r_via_mod:Routine)
WITH coalesce(direct_r, r_via_mod) AS r
WHERE r IS NOT NULL AND r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
""".strip(), {"qn": owner_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=owner_qn, from_label=owner_label,
            to_ref=r["ref"], to_label="Routine",
            to_name=r.get("name") or "",
            relationship_type="OWNER_BRIDGE", owner_step=True,
            to_owner_ref=r.get("owner_qn"),
        )
        for r in rows if r.get("ref")
    ]


def _expand_binds_to_downstream(
    loader: Any, ctrl_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    rows = _run_query(loader, """
MATCH (fc:FormControl {qualified_name: $qn})-[:BINDS_TO]->(target)
WHERE (target:Attribute OR target:Dimension OR target:Resource
    OR target:FormAttribute OR target:MetadataObject)
RETURN labels(target)[0] AS label, target.qualified_name AS ref, target.name AS name
""".strip(), {"qn": ctrl_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=ctrl_qn, from_label="FormControl",
            to_ref=r["ref"], to_label=r["label"],
            to_name=r.get("name") or "",
            relationship_type="BINDS_TO", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_binds_to_upstream(
    loader: Any, target_ref: str, target_label: str,
    project_name: str, config_name: Optional[str],
) -> List[TraversalStep]:
    qn_prefix = (project_name + "/" + config_name + "/") if config_name else (project_name + "/")
    rows = _run_query(loader, """
MATCH (fc:FormControl)-[:BINDS_TO]->(target {qualified_name: $qn})
WHERE fc.qualified_name STARTS WITH $qn_prefix
RETURN fc.qualified_name AS ref, fc.name AS name
""".strip(), {"qn": target_ref, "config_name": config_name, "qn_prefix": qn_prefix}, project_name)
    return [
        TraversalStep(
            from_ref=target_ref, from_label=target_label,
            to_ref=r["ref"], to_label="FormControl",
            to_name=r.get("name") or "",
            relationship_type="BINDS_TO", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_links_to_command_downstream(
    loader: Any, ctrl_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    rows = _run_query(loader, """
MATCH (fc:FormControl {qualified_name: $qn})-[:LINKS_TO_COMMAND]->(cmd:Command)
RETURN cmd.qualified_name AS ref, cmd.name AS name
""".strip(), {"qn": ctrl_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=ctrl_qn, from_label="FormControl",
            to_ref=r["ref"], to_label="Command",
            to_name=r.get("name") or "",
            relationship_type="LINKS_TO_COMMAND", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_links_to_command_upstream(
    loader: Any, cmd_qn: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    qn_prefix = (project_name + "/" + config_name + "/") if config_name else (project_name + "/")
    rows = _run_query(loader, """
MATCH (fc:FormControl)-[:LINKS_TO_COMMAND]->(cmd:Command {qualified_name: $qn})
WHERE fc.qualified_name STARTS WITH $qn_prefix
RETURN fc.qualified_name AS ref, fc.name AS name
""".strip(), {"qn": cmd_qn, "config_name": config_name, "qn_prefix": qn_prefix}, project_name)
    return [
        TraversalStep(
            from_ref=cmd_qn, from_label="Command",
            to_ref=r["ref"], to_label="FormControl",
            to_name=r.get("name") or "",
            relationship_type="LINKS_TO_COMMAND", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_has_handler_downstream(
    loader: Any, src_qn: str, src_label: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg_r = _cfg_and("r", config_name)
    rows = _run_query(loader, f"""
MATCH (src:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:HAS_HANDLER]->(r:Routine)
WHERE r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
""".strip(), {"qn": src_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=src_qn, from_label=src_label,
            to_ref=r["ref"], to_label="Routine",
            to_name=r.get("name") or "",
            relationship_type="HAS_HANDLER", owner_step=False,
            to_owner_ref=r.get("owner_qn"),
        )
        for r in rows if r.get("ref")
    ]


def _expand_has_handler_upstream(
    loader: Any, routine_id: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg_src = (
        " AND src.qualified_name STARTS WITH ($project_name + '/' + $config_name + '/')"
        if config_name else ""
    )
    rows = _run_query(loader, f"""
MATCH (src)-[:HAS_HANDLER]->(r:Routine {{id: $routine_id, project_name: $project_name}})
WHERE (src:UrlMethod OR src:FormEventAction OR src:Command){cfg_src}
RETURN src.qualified_name AS ref, labels(src)[0] AS label, src.name AS name
""".strip(), {"routine_id": routine_id, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=routine_id, from_label="Routine",
            to_ref=r["ref"], to_label=r.get("label") or "",
            to_name=r.get("name") or "",
            relationship_type="HAS_HANDLER", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_uses_handler_downstream(
    loader: Any, src_qn: str, src_label: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg_r = _cfg_and("r", config_name)
    rows = _run_query(loader, f"""
MATCH (src:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:USES_HANDLER]->(r:Routine)
WHERE r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
""".strip(), {"qn": src_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=src_qn, from_label=src_label,
            to_ref=r["ref"], to_label="Routine",
            to_name=r.get("name") or "",
            relationship_type="USES_HANDLER", owner_step=False,
            to_owner_ref=r.get("owner_qn"),
        )
        for r in rows if r.get("ref")
    ]


def _expand_uses_handler_upstream(
    loader: Any, routine_id: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    cfg_src = (
        " AND src.qualified_name STARTS WITH ($project_name + '/' + $config_name + '/')"
        if config_name else ""
    )
    rows = _run_query(loader, f"""
MATCH (src)-[:USES_HANDLER]->(r:Routine {{id: $routine_id, project_name: $project_name}})
WHERE true{cfg_src}
RETURN src.qualified_name AS ref, labels(src)[0] AS label, src.name AS name
""".strip(), {"routine_id": routine_id, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=routine_id, from_label="Routine",
            to_ref=r["ref"], to_label=r.get("label") or "",
            to_name=r.get("name") or "",
            relationship_type="USES_HANDLER", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


# ---------------------------------------------------------------------------
# Form-container flattened expansion (regular Form + common form MetadataObject)
# ---------------------------------------------------------------------------

def _is_form_container(label: str, qn: str) -> bool:
    """A form container is a regular Form node or a common form (MetadataObject under
    /ОбщиеФормы/). Both carry HAS_CONTROL/HAS_EVENT/HAS_COMMAND; ordinary MetadataObjects
    must NOT enter the form-only branches."""
    return label == "Form" or (label == "MetadataObject" and "/ОбщиеФормы/" in qn)


def _expand_form_binds_to_downstream(
    loader: Any, owner_qn: str, owner_label: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    """Flatten owner -> controls -> BINDS_TO target into a direct owner -> target step."""
    rows = _run_query(loader, f"""
MATCH (owner:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:HAS_CONTROL]->(:FormControl)-[:HAS_CHILD*0..]->(fc:FormControl)-[:BINDS_TO]->(target)
WHERE (target:Attribute OR target:Dimension OR target:Resource
    OR target:FormAttribute OR target:MetadataObject)
RETURN labels(target)[0] AS label, target.qualified_name AS ref, target.name AS name
""".strip(), {"qn": owner_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=owner_qn, from_label=owner_label,
            to_ref=r["ref"], to_label=r["label"],
            to_name=r.get("name") or "",
            relationship_type="BINDS_TO", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_form_links_to_command_downstream(
    loader: Any, owner_qn: str, owner_label: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    """Flatten owner -> controls -> LINKS_TO_COMMAND into a direct owner -> Command step."""
    rows = _run_query(loader, f"""
MATCH (owner:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:HAS_CONTROL]->(:FormControl)-[:HAS_CHILD*0..]->(fc:FormControl)-[:LINKS_TO_COMMAND]->(cmd:Command)
RETURN cmd.qualified_name AS ref, cmd.name AS name
""".strip(), {"qn": owner_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=owner_qn, from_label=owner_label,
            to_ref=r["ref"], to_label="Command",
            to_name=r.get("name") or "",
            relationship_type="LINKS_TO_COMMAND", owner_step=False,
        )
        for r in rows if r.get("ref")
    ]


def _expand_form_handler_downstream(
    loader: Any, owner_qn: str, owner_label: str, rel: str,
    project_name: str, config_name: Optional[str],
) -> List[TraversalStep]:
    """Flatten a form container's handlers (form-level events, control-level events, form
    commands) into direct owner -> Routine steps for rel in {HAS_HANDLER, USES_HANDLER}."""
    cfg_r = _cfg_and("r", config_name)
    rows = _run_query(loader, f"""
MATCH (owner:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:HAS_EVENT]->(:FormEvent)-[:HAS_EVENT_ACTION]->(:FormEventAction)-[:{rel}]->(r:Routine)
WHERE r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
UNION
MATCH (owner:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:HAS_CONTROL]->(:FormControl)-[:HAS_CHILD*0..]->(:FormControl)-[:HAS_EVENT]->(:FormEvent)-[:HAS_EVENT_ACTION]->(:FormEventAction)-[:{rel}]->(r:Routine)
WHERE r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
UNION
MATCH (owner:{QN_LABEL_EXPR} {{qualified_name: $qn}})-[:HAS_COMMAND]->(:Command)-[:{rel}]->(r:Routine)
WHERE r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
""".strip(), {"qn": owner_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=owner_qn, from_label=owner_label,
            to_ref=r["ref"], to_label="Routine",
            to_name=r.get("name") or "",
            relationship_type=rel, owner_step=False,
            to_owner_ref=r.get("owner_qn"),
        )
        for r in rows if r.get("ref")
    ]


def _expand_formevent_handler_downstream(
    loader: Any, fe_qn: str, rel: str, project_name: str, config_name: Optional[str]
) -> List[TraversalStep]:
    """Flatten FormEvent -> FormEventAction -> Routine into a direct FormEvent -> Routine step
    for rel in {HAS_HANDLER, USES_HANDLER}."""
    cfg_r = _cfg_and("r", config_name)
    rows = _run_query(loader, f"""
MATCH (fe:FormEvent {{qualified_name: $qn}})-[:HAS_EVENT_ACTION]->(:FormEventAction)-[:{rel}]->(r:Routine)
WHERE r.project_name = $project_name{cfg_r}
RETURN r.id AS ref, r.name AS name, r.owner_qn AS owner_qn
""".strip(), {"qn": fe_qn, "config_name": config_name}, project_name)
    return [
        TraversalStep(
            from_ref=fe_qn, from_label="FormEvent",
            to_ref=r["ref"], to_label="Routine",
            to_name=r.get("name") or "",
            relationship_type=rel, owner_step=False,
            to_owner_ref=r.get("owner_qn"),
        )
        for r in rows if r.get("ref")
    ]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _expand(
    loader: Any,
    cur_ref: str,
    cur_label: str,
    direction: str,
    rel_types: List[str],
    project_name: str,
    config_name: Optional[str],
) -> List[TraversalStep]:
    """Expand one BFS frontier node across all requested relationship types."""
    results: List[TraversalStep] = []
    bsl_ok = settings.load_bsl_signatures

    for rel_type in rel_types:
        if rel_type in _BSL_RELS and not bsl_ok:
            continue

        if rel_type == "USED_IN":
            if cur_label == "MetadataObject":
                if direction in ("downstream", "both"):
                    results += _expand_used_in_downstream(loader, cur_ref, project_name, config_name)
                if direction in ("upstream", "both"):
                    results += _expand_used_in_upstream(loader, cur_ref, project_name, config_name)
            elif cur_label in _ELEMENT_LABELS and direction in ("upstream", "both"):
                # Bridge to owner MetadataObject first; real expansion happens in next BFS iteration
                owner_qn = _owner_qn_from_element_qn(cur_ref)
                results.append(TraversalStep(
                    from_ref=cur_ref, from_label=cur_label,
                    to_ref=owner_qn, to_label="MetadataObject",
                    to_name="",
                    relationship_type="OWNER_BRIDGE", owner_step=True,
                ))

        elif rel_type == "DO_MOVEMENTS_IN":
            if cur_label == "MetadataObject":
                if direction in ("downstream", "both"):
                    results += _expand_do_movements_downstream(loader, cur_ref, project_name, config_name)
                if direction in ("upstream", "both"):
                    results += _expand_do_movements_upstream(loader, cur_ref, project_name, config_name)
            elif cur_label in _ELEMENT_LABELS and direction in ("upstream", "both"):
                owner_qn = _owner_qn_from_element_qn(cur_ref)
                results.append(TraversalStep(
                    from_ref=cur_ref, from_label=cur_label,
                    to_ref=owner_qn, to_label="MetadataObject",
                    to_name="",
                    relationship_type="OWNER_BRIDGE", owner_step=True,
                ))

        elif rel_type == "CALLS":
            if cur_label == "Routine":
                results += _expand_calls_from_routine(loader, cur_ref, direction, project_name, config_name)
            elif cur_label in ("MetadataObject", "Command", "Form"):
                # Form container (incl. common form MetadataObject) owns a form module the same way.
                results += _expand_calls_from_owner(loader, cur_ref, cur_label, project_name, config_name)

        elif rel_type == "BINDS_TO":
            if cur_label == "FormControl" and direction in ("downstream", "both"):
                results += _expand_binds_to_downstream(loader, cur_ref, project_name, config_name)
            if _is_form_container(cur_label, cur_ref) and direction in ("downstream", "both"):
                results += _expand_form_binds_to_downstream(loader, cur_ref, cur_label, project_name, config_name)
            if direction in ("upstream", "both"):
                results += _expand_binds_to_upstream(loader, cur_ref, cur_label, project_name, config_name)

        elif rel_type == "LINKS_TO_COMMAND":
            if cur_label == "FormControl" and direction in ("downstream", "both"):
                results += _expand_links_to_command_downstream(loader, cur_ref, project_name, config_name)
            elif _is_form_container(cur_label, cur_ref) and direction in ("downstream", "both"):
                results += _expand_form_links_to_command_downstream(loader, cur_ref, cur_label, project_name, config_name)
            elif cur_label == "Command" and direction in ("upstream", "both"):
                results += _expand_links_to_command_upstream(loader, cur_ref, project_name, config_name)

        elif rel_type == "HAS_HANDLER":
            if cur_label in ("UrlMethod", "FormEventAction", "Command") and direction in ("downstream", "both"):
                results += _expand_has_handler_downstream(loader, cur_ref, cur_label, project_name, config_name)
            elif _is_form_container(cur_label, cur_ref) and direction in ("downstream", "both"):
                results += _expand_form_handler_downstream(loader, cur_ref, cur_label, "HAS_HANDLER", project_name, config_name)
            elif cur_label == "FormEvent" and direction in ("downstream", "both"):
                results += _expand_formevent_handler_downstream(loader, cur_ref, "HAS_HANDLER", project_name, config_name)
            elif cur_label == "Routine" and direction in ("upstream", "both"):
                results += _expand_has_handler_upstream(loader, cur_ref, project_name, config_name)

        elif rel_type == "USES_HANDLER":
            if _is_form_container(cur_label, cur_ref) and direction in ("downstream", "both"):
                results += _expand_form_handler_downstream(loader, cur_ref, cur_label, "USES_HANDLER", project_name, config_name)
            elif cur_label == "FormEvent" and direction in ("downstream", "both"):
                results += _expand_formevent_handler_downstream(loader, cur_ref, "USES_HANDLER", project_name, config_name)
            elif direction in ("downstream", "both"):
                results += _expand_uses_handler_downstream(loader, cur_ref, cur_label, project_name, config_name)
            if direction in ("upstream", "both"):
                results += _expand_uses_handler_upstream(loader, cur_ref, project_name, config_name)

    # Deduplicate by (to_ref, relationship_type, owner_step) to prevent duplicate bridges
    seen: Set[tuple] = set()
    unique: List[TraversalStep] = []
    for s in results:
        key = (s.to_ref, s.relationship_type, s.owner_step)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# BFS engine
# ---------------------------------------------------------------------------

def _build_path(
    start_ref: str,
    start_label: str,
    steps: List[TraversalStep],
    depth: int,
) -> Optional[DependencyPath]:
    real_steps = [s for s in steps if not s.owner_step]
    if not real_steps:
        return None
    end_step = real_steps[-1]
    relationship_chain = [s.relationship_type for s in real_steps]

    end_owner_ref: Optional[str] = None
    if end_step.to_label in _ELEMENT_LABELS:
        end_owner_ref = _owner_qn_from_element_qn(end_step.to_ref)
    elif end_step.to_label == "Routine" and end_step.to_owner_ref:
        end_owner_ref = end_step.to_owner_ref

    start_name = start_ref.split("/")[-1] if "/" in start_ref else start_ref
    end_name_display = end_step.to_name or (
        end_step.to_ref.split("/")[-1] if "/" in end_step.to_ref else end_step.to_ref
    )
    chain_str = ", ".join(relationship_chain)
    path_display = f"depth={depth}: {start_name} --[{chain_str}]--> {end_name_display}"

    return DependencyPath(
        depth=depth,
        start_ref=start_ref,
        start_label=start_label,
        end_ref=end_step.to_ref,
        end_label=end_step.to_label,
        end_name=end_step.to_name,
        end_owner_ref=end_owner_ref,
        relationship_chain=relationship_chain,
        steps=steps,
        path_display=path_display,
    )


def traverse(
    loader: Any,
    start_ref: str,
    start_label: str,
    direction: str,
    rel_types: List[str],
    max_depth: int,
    project_name: str,
    config_name: Optional[str],
    max_paths: Optional[int] = None,
) -> List[DependencyPath]:
    frontier: List[Tuple[str, str, List[TraversalStep], int]] = [
        (start_ref, start_label, [], 0)
    ]
    completed: List[DependencyPath] = []
    seen_sigs: Set[tuple] = set()
    _owner_label_cache: Dict[str, str] = {}

    while frontier:
        next_frontier: List[Tuple[str, str, List[TraversalStep], int]] = []
        for cur_ref, cur_label, steps, depth in frontier:
            if depth >= max_depth:
                continue

            hops = _expand(loader, cur_ref, cur_label, direction, rel_types, project_name, config_name)
            visited_in_path: Set[str] = {s.to_ref for s in steps}
            visited_in_path.add(start_ref)

            for hop in hops:
                if hop.to_ref in visited_in_path:
                    continue
                new_steps = steps + [hop]

                if hop.owner_step:
                    next_frontier.append((hop.to_ref, hop.to_label, new_steps, depth))
                else:
                    new_depth = depth + 1
                    path = _build_path(start_ref, start_label, new_steps, new_depth)
                    if path:
                        sig = _path_signature(path)
                        if sig not in seen_sigs:
                            seen_sigs.add(sig)
                            completed.append(path)
                            if max_paths is not None and len(completed) >= max_paths:
                                return completed

                    if new_depth < max_depth:
                        bridge = _make_owner_bridge(hop, loader, _owner_label_cache)
                        bridge_refs = visited_in_path | {hop.to_ref}
                        if hop.to_label == "Routine":
                            # Routine stays in frontier for precise CALLS chain continuity
                            next_frontier.append((hop.to_ref, hop.to_label, new_steps, new_depth))
                            # Also add owner bridge for cross-family transitions
                            if bridge and bridge.to_ref not in bridge_refs:
                                next_frontier.append(
                                    (bridge.to_ref, bridge.to_label, new_steps + [bridge], new_depth)
                                )
                        elif bridge and bridge.to_ref not in bridge_refs:
                            next_frontier.append(
                                (bridge.to_ref, bridge.to_label, new_steps + [bridge], new_depth)
                            )
                        else:
                            next_frontier.append((hop.to_ref, hop.to_label, new_steps, new_depth))

        frontier = next_frontier

    return completed


# ---------------------------------------------------------------------------
# Deduplication & formatting
# ---------------------------------------------------------------------------

def _path_signature(path: DependencyPath) -> tuple:
    return tuple((s.from_ref, s.to_ref, s.relationship_type) for s in path.steps)


def dedup_paths(paths: List[DependencyPath]) -> List[DependencyPath]:
    seen: Set[tuple] = set()
    result: List[DependencyPath] = []
    for p in paths:
        sig = _path_signature(p)
        if sig not in seen:
            seen.add(sig)
            result.append(p)
    return result


def path_to_text_row(path: DependencyPath) -> Dict[str, Any]:
    return {"depth": path.depth, "path": path.path_display}
