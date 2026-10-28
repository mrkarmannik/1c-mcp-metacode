"""
Callsites resolver for BSL code.

Resolves function/procedure calls between routines, handling:
- Direct calls (within same module)
- Qualified calls (CommonModule.Procedure)
- Dynamic calls (with runtime evaluation)
- Notification callbacks
- Directive compatibility (&НаКлиенте, &НаСервере, etc.)
- Arity-based overload resolution
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Mapping
from collections import defaultdict
import logging

from graphdb.category_canon import canon_category

logger = logging.getLogger(__name__)


class CallsitesResolver:
    """Resolves BSL callsites to actual routine targets"""

    def __init__(self):
        """Initialize the callsites resolver"""
        pass

    def resolve_calls(
        self,
        routines_indexes: List[Dict[str, Any]],
        callsites: List[Dict[str, Any]],
        project_name: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Resolve all callsites to their target routines.

        Args:
            routines_indexes: Lightweight routine indexes (id, name, params, directives)
            callsites: List of callsite dictionaries
            project_name: Project name for the call rows

        Returns:
            Tuple of (call_rows, sorted_caller_ids)
        """
        return self.resolve_prepared(self.prepare(routines_indexes), callsites, project_name)

    def prepare(self, routines_indexes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Строит индексы корпуса один раз — для резолва callsites порциями."""
        arity_by_rid = self._build_arity_index(routines_indexes)
        common_owner_by_name, commons_sig_by_owner_and_name = \
            self._build_common_module_lookups(routines_indexes, arity_by_rid)
        return {
            "routines_count": len(routines_indexes),
            "routine_by_id": self._build_routine_index(routines_indexes),
            "module_id_by_rid": self._build_module_index(routines_indexes),
            "namesig_by_module": self._build_module_lookups(routines_indexes, arity_by_rid),
            "common_owner_by_name": common_owner_by_name,
            "commons_sig_by_owner_and_name": commons_sig_by_owner_and_name,
            "mgr_by_cat_obj_name": self._build_manager_lookups(routines_indexes, arity_by_rid),
        }

    def resolve_prepared(
        self,
        prepared: Dict[str, Any],
        callsites: List[Dict[str, Any]],
        project_name: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Как `resolve_calls`, но по индексам из `prepare`."""
        logger.info("Resolving %d callsites against %d routines...", len(callsites), prepared["routines_count"])

        routine_by_id = prepared["routine_by_id"]
        module_id_by_rid = prepared["module_id_by_rid"]
        namesig_by_module = prepared["namesig_by_module"]
        common_owner_by_name = prepared["common_owner_by_name"]
        commons_sig_by_owner_and_name = prepared["commons_sig_by_owner_and_name"]
        mgr_by_cat_obj_name = prepared["mgr_by_cat_obj_name"]

        # Resolve callsites
        callers_set: Set[str] = set()
        edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for cs in callsites:
            caller_id = cs.get("caller_id")
            if not caller_id:
                continue

            callers_set.add(caller_id)
            caller = routine_by_id.get(caller_id) or {}
            caller_dirs = caller.get("directives", [])
            caller_mid = module_id_by_rid.get(caller_id)
            caller_config_name = self._config_seg_from_owner_qn(caller.get("owner_qn"))
            kind = (cs.get("kind") or "direct").lower()
            line_no = cs.get("line")
            args_cnt = cs.get("args_count")
            raw_name = cs.get("name") or cs.get("name_literal") or ""
            name_low = self._name_low(raw_name)

            callee_id, edge_kind = self._resolve_single_callsite(
                kind=kind,
                name_low=name_low,
                qualifier=cs.get("qualifier"),
                qualifier_parts=cs.get("qualifier_parts"),
                args_cnt=args_cnt,
                caller_mid=caller_mid,
                caller_dirs=caller_dirs,
                caller_config_name=caller_config_name,
                namesig_by_module=namesig_by_module,
                common_owner_by_name=common_owner_by_name,
                commons_sig_by_owner_and_name=commons_sig_by_owner_and_name,
                mgr_by_cat_obj_name=mgr_by_cat_obj_name,
            )

            if callee_id:
                self._add_edge(edges, caller_id, callee_id, edge_kind, line_no)

        # Build call rows
        call_rows = self._build_call_rows(edges, project_name)
        sorted_callers = sorted(callers_set)

        logger.info("Resolved %d edges from %d callers", len(call_rows), len(sorted_callers))

        return call_rows, sorted_callers

    def _build_routine_index(self, routines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build routine_id -> routine dictionary"""
        return {r.get("id"): r for r in routines if r.get("id")}

    @staticmethod
    def _module_key(r: Dict[str, Any]) -> Optional[str]:
        """Ключ модуля процедуры для локальных вызовов.

        Процедуры общих модулей идут без узла Module (`module_id` = None, см. bsl_signature_scanner),
        поэтому их модуль — сам общий модуль, `owner_qn`. Без этого вызов без префикса внутри общего
        модуля не находил цель.
        """
        module_id = r.get("module_id")
        if module_id:
            return module_id
        owner_qn = r.get("owner_qn")
        return f"owner:{owner_qn}" if owner_qn else None

    def _build_module_index(self, routines: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """Build routine_id -> module key dictionary (см. `_module_key`)"""
        return {r.get("id"): self._module_key(r) for r in routines if r.get("id")}

    def _build_arity_index(self, routines: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        """
        Build routine_id -> (min_arity, max_arity) dictionary.

        Analyzes parameters to determine min/max argument counts.
        """
        arity_by_rid: Dict[str, Tuple[int, int]] = {}

        for r in routines:
            rid = r.get("id")
            if not rid:
                continue

            params = r.get("params_json") or []
            min_a = 0

            try:
                for p in params:
                    if not isinstance(p, dict):
                        min_a += 1
                        continue

                    default_present = bool(p.get("default_present"))
                    markers = [str(m) for m in (p.get("markers_raw") or []) if isinstance(m, str)]
                    is_optional = default_present or any(("необязатель" in str(m).casefold()) for m in markers)

                    if not is_optional:
                        min_a += 1

                max_a = len(params)
            except Exception:
                min_a = 0
                max_a = len(params) if isinstance(params, list) else 0

            arity_by_rid[rid] = (min_a, max_a)

        return arity_by_rid

    def _build_module_lookups(
        self,
        routines: List[Dict[str, Any]],
        arity_by_rid: Dict[str, Tuple[int, int]]
    ) -> Mapping[str, Mapping[str, List[Tuple[str, int, int, List[str]]]]]:  # namesig_by_module
        """
        Build the module-based signature lookup.

        Returns:
            namesig_by_module: module key -> name_low -> [(rid, min_a, max_a, directives), ...]
        """
        namesig_by_module: Mapping[str, Mapping[str, List[Tuple[str, int, int, List[str]]]]] = \
            defaultdict(lambda: defaultdict(list))

        for r in routines:
            rid = r.get("id")
            if not rid:
                continue

            module_id = self._module_key(r)
            if not module_id:
                continue

            name_low = self._name_low(r.get("name") or "")
            directives = r.get("directives", [])
            min_a, max_a = arity_by_rid.get(rid, (0, 0))

            namesig_by_module[module_id][name_low].append((rid, min_a, max_a, directives))

        return namesig_by_module

    def _build_common_module_lookups(
        self,
        routines: List[Dict[str, Any]],
        arity_by_rid: Dict[str, Tuple[int, int]]
    ) -> Tuple[
        Dict[str, List[str]],  # common_owner_by_name: name -> [owner_qn, ...]
        Mapping[str, Mapping[str, List[Tuple[str, int, int, List[str]]]]]  # commons_sig_by_owner_and_name
    ]:
        """
        Build CommonModule-based lookup structures.

        Returns:
            Tuple of (common_owner_by_name, commons_sig_by_owner_and_name)
        """
        common_owner_by_name: Dict[str, List[str]] = defaultdict(list)
        commons_sig_by_owner_and_name: Mapping[str, Mapping[str, List[Tuple[str, int, int, List[str]]]]] = \
            defaultdict(lambda: defaultdict(list))

        for r in routines:
            rid = r.get("id")
            if not rid:
                continue

            module_id = r.get("module_id")
            if module_id:
                continue  # Not a CommonModule routine

            # CommonModule routine (no module_id)
            owner_qn = r.get("owner_qn") or ""
            if not owner_qn:
                continue

            cm_name = owner_qn.split("/")[-1] if owner_qn else ""
            cm_low = self._name_low(cm_name)

            if cm_low and owner_qn not in common_owner_by_name[cm_low]:
                common_owner_by_name[cm_low].append(owner_qn)

            # Only exported routines are accessible
            if not r.get("export", False):
                continue

            name_low = self._name_low(r.get("name") or "")
            directives = r.get("directives", [])
            min_a, max_a = arity_by_rid.get(rid, (0, 0))

            commons_sig_by_owner_and_name[owner_qn][name_low].append((rid, min_a, max_a, directives))

        return common_owner_by_name, commons_sig_by_owner_and_name

    def _build_manager_lookups(
        self,
        routines: List[Dict[str, Any]],
        arity_by_rid: Dict[str, Tuple[int, int]]
    ) -> Mapping[Tuple[str, str, str], List[Tuple[str, int, int, List[str], str]]]:
        """
        Build lookup for exported routines in ManagerModule / ValueManagerModule.

        Key: (canon_category, object_name_low, routine_name_low)
        Value: list of (routine_id, min_arity, max_arity, directives, config_name)
        """
        mgr: Mapping[Tuple[str, str, str], List[Tuple[str, int, int, List[str], str]]] = defaultdict(list)

        for r in routines:
            module_type = r.get("module_type")
            if module_type not in ("ManagerModule", "ValueManagerModule"):
                continue
            if not r.get("export", False):
                continue

            owner_qn = r.get("owner_qn") or ""
            parts = owner_qn.split("/")
            if len(parts) < 4:
                continue
            _project_seg, config_seg, cat_seg, obj_seg = parts[0], parts[1], parts[2], parts[3]

            canons = canon_category(cat_seg)
            if not canons:
                continue
            canon_cat = canons[0]

            rid = r.get("id")
            if not rid:
                continue

            obj_low = self._name_low(obj_seg)
            name_low = self._name_low(r.get("name") or "")
            min_a, max_a = arity_by_rid.get(rid, (0, 0))
            directives = r.get("directives", []) or []

            mgr[(canon_cat, obj_low, name_low)].append(
                (rid, min_a, max_a, directives, config_seg)
            )

        return mgr

    def _pick_manager_candidate(
        self,
        cands: List[Tuple[str, int, int, List[str], str]],
        caller_config_name: Optional[str],
        caller_dirs: List[str],
        args_cnt: Optional[int]
    ) -> Optional[str]:
        """
        Pick a single manager-routine candidate.

        Unlike _pick_best_by_arity_and_dir, this returns a candidate ONLY when
        the post-filter result is unambiguous (exactly one match). Ambiguity
        is treated as "do not create edge" to avoid false-positive base/ext
        links.
        """
        if not cands:
            return None

        # 1. Prefer same config as caller.
        if caller_config_name:
            same_config = [c for c in cands if c[4] == caller_config_name]
            if len(same_config) == 1:
                return same_config[0][0]

        # 2. Filter by arity + directive compatibility (mirrors the first pass
        #    in _pick_best_by_arity_and_dir, but works on the full 5-tuple).
        filtered: List[Tuple[str, int, int, List[str], str]] = []
        for rid, min_a, max_a, cdirs, cfg in cands:
            if args_cnt is None or (min_a <= args_cnt <= max_a):
                if self._dir_compatible(caller_dirs, cdirs):
                    filtered.append((rid, min_a, max_a, cdirs, cfg))

        # 3. Fallback to arity-only (mirrors second pass in helper).
        if not filtered:
            for rid, min_a, max_a, cdirs, cfg in cands:
                if args_cnt is None or (min_a <= args_cnt <= max_a):
                    filtered.append((rid, min_a, max_a, cdirs, cfg))

        # 4. Length check: only emit edge on a single unambiguous match.
        if len(filtered) == 1:
            return filtered[0][0]
        return None

    @staticmethod
    def _config_seg_from_owner_qn(owner_qn: Optional[str]) -> Optional[str]:
        """Return the config segment from an owner_qn (project/config/...)."""
        if not owner_qn:
            return None
        parts = owner_qn.split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return None

    def _resolve_single_callsite(
        self,
        kind: str,
        name_low: str,
        qualifier: Optional[str],
        qualifier_parts: Optional[List[str]],
        args_cnt: Optional[int],
        caller_mid: Optional[str],
        caller_dirs: List[str],
        caller_config_name: Optional[str],
        namesig_by_module: Mapping[str, Mapping[str, List[Tuple[str, int, int, List[str]]]]],
        common_owner_by_name: Dict[str, List[str]],
        commons_sig_by_owner_and_name: Mapping[str, Mapping[str, List[Tuple[str, int, int, List[str]]]]],
        mgr_by_cat_obj_name: Mapping[Tuple[str, str, str], List[Tuple[str, int, int, List[str], str]]]
    ) -> Tuple[Optional[str], str]:
        """
        Resolve a single callsite to a routine ID.

        Returns:
            Tuple of (callee_id, edge_kind)
        """
        callee_id: Optional[str] = None
        edge_kind = kind

        if kind == "direct":
            if caller_mid:
                cands = namesig_by_module.get(caller_mid, {}).get(name_low) or []
                callee_id = self._pick_best_by_arity_and_dir(cands, None, caller_dirs)

        elif kind == "qualified":
            qualifier_low = self._name_low(qualifier or "")
            if qualifier_parts:
                parts = qualifier_parts
            elif qualifier:
                # Backward-compatible fallback for callsite payloads emitted
                # before qualifier_parts was added (e.g. queued from an older
                # worker version): split the scalar qualifier on '.'.
                parts = qualifier.split(".")
            else:
                parts = []
            if qualifier_low in ("этотобъект", "thisobject"):
                # Method on current module
                if caller_mid:
                    cands = namesig_by_module.get(caller_mid, {}).get(name_low) or []
                    callee_id = self._pick_best_by_arity_and_dir(cands, None, caller_dirs)
                    edge_kind = "method"
            elif len(parts) == 2:
                # Manager call: Category.Object.Method
                cat_canons = canon_category(parts[0])
                if cat_canons:
                    canon_cat = cat_canons[0]
                    obj_low = self._name_low(parts[1])
                    cands = mgr_by_cat_obj_name.get((canon_cat, obj_low, name_low)) or []
                    callee_id = self._pick_manager_candidate(
                        cands, caller_config_name, caller_dirs, args_cnt
                    )
                    edge_kind = "qualified"
            else:
                owner_qns = common_owner_by_name.get(qualifier_low) or []
                if owner_qns:
                    cands = []
                    for oqn in owner_qns:
                        cands.extend(commons_sig_by_owner_and_name.get(oqn, {}).get(name_low) or [])
                    callee_id = self._pick_best_by_arity_and_dir(cands, args_cnt, caller_dirs)
                    edge_kind = "qualified"

        elif kind == "dynamic":
            qualifier_low = self._name_low(qualifier or "")
            if qualifier_low in ("этотобъект", "thisobject"):
                if caller_mid:
                    cands = namesig_by_module.get(caller_mid, {}).get(name_low) or []
                    callee_id = self._pick_best_by_arity_and_dir(cands, args_cnt, caller_dirs)
            else:
                owner_qns = common_owner_by_name.get(qualifier_low) if qualifier_low else []
                if owner_qns:
                    cands = []
                    for oqn in owner_qns:
                        cands.extend(commons_sig_by_owner_and_name.get(oqn, {}).get(name_low) or [])
                    callee_id = self._pick_best_by_arity_and_dir(cands, args_cnt, caller_dirs)
                else:
                    # Treat as local dynamic call
                    if caller_mid:
                        cands = namesig_by_module.get(caller_mid, {}).get(name_low) or []
                        callee_id = self._pick_best_by_arity_and_dir(cands, args_cnt, caller_dirs)
            edge_kind = "dynamic"

        elif kind == "notification":
            # Assume callback on current module (ThisObject)
            if caller_mid:
                cands = namesig_by_module.get(caller_mid, {}).get(name_low) or []
                callee_id = self._pick_best_by_arity_and_dir(cands, None, caller_dirs)
            edge_kind = "notification"

        return callee_id, edge_kind

    def _pick_best_by_arity_and_dir(
        self,
        cands: List[Tuple[str, int, int, List[str]]],
        args_cnt: Optional[int],
        caller_dirs: List[str]
    ) -> Optional[str]:
        """
        Pick the best candidate based on arity and directive compatibility.

        Args:
            cands: List of (routine_id, min_arity, max_arity, directives)
            args_cnt: Number of arguments (None if unknown)
            caller_dirs: Caller's directives

        Returns:
            Best matching routine_id or None
        """
        if not cands:
            return None

        # First filter: arity + directive compatibility
        filtered = []
        for rid, min_a, max_a, cdirs in cands:
            if args_cnt is None or (min_a <= args_cnt <= max_a):
                if self._dir_compatible(caller_dirs, cdirs):
                    filtered.append((rid, min_a, max_a, cdirs))

        # Second filter: arity only (fallback)
        if not filtered:
            for rid, min_a, max_a, cdirs in cands:
                if args_cnt is None or (min_a <= args_cnt <= max_a):
                    filtered.append((rid, min_a, max_a, cdirs))

        # Final fallback: any candidate
        if not filtered:
            filtered = cands[:]

        # Choose candidate with minimal min_arity, tie-breaker: minimal max_arity
        filtered.sort(key=lambda t: (t[1], t[2]))
        return filtered[0][0] if filtered else None

    def _classify_directives(self, directives: Optional[List[str]]) -> Set[str]:
        """
        Classify directives into execution context categories.

        Returns set of: {"client", "server", "server_no_context", "external_connection", "neutral"}
        """
        if not directives:
            return {"neutral"}

        classes = set()
        for d in directives:
            s = d.casefold()
            if "наклиенте" in s or "atclient" in s:
                classes.add("client")
            elif "насервере" in s or "atserver" in s:
                if "безконтекста" in s or "nocontext" in s:
                    classes.add("server_no_context")
                else:
                    classes.add("server")
            elif "навнешнемсоединении" in s or "atexternalconnection" in s:
                classes.add("external_connection")

        return classes if classes else {"neutral"}

    def _dir_compatible(self, caller_dirs: Optional[List[str]], callee_dirs: Optional[List[str]]) -> bool:
        """
        Check if caller and callee directives are compatible.

        Args:
            caller_dirs: Caller's directives
            callee_dirs: Callee's directives

        Returns:
            True if compatible
        """
        caller_classes = self._classify_directives(caller_dirs)
        callee_classes = self._classify_directives(callee_dirs)

        # If callee has both client+server directives, always compatible
        if {"client", "server"}.issubset(callee_classes):
            return True

        # Neutral context is compatible with anything
        if "neutral" in callee_classes or "neutral" in caller_classes:
            return True

        # server_no_context is compatible with any server context
        if "server_no_context" in callee_classes and ("server" in caller_classes or "client" in caller_classes):
            return True

        # Check intersection of contexts
        return len(caller_classes & callee_classes) > 0

    def _add_edge(
        self,
        edges: Dict[Tuple[str, str, str], Dict[str, Any]],
        caller_id: str,
        callee_id: str,
        kind: str,
        line_no: Optional[int]
    ):
        """Add or update an edge in the edges dictionary"""
        key = (caller_id, callee_id, kind)
        e = edges.get(key)

        if not e:
            edges[key] = {"count": 1, "lines": [line_no] if line_no else []}
        else:
            e["count"] += 1
            if line_no:
                L = e["lines"]
                if len(L) < 50 and line_no not in L:
                    L.append(line_no)

    def _build_call_rows(
        self,
        edges: Dict[Tuple[str, str, str], Dict[str, Any]],
        project_name: str
    ) -> List[Dict[str, Any]]:
        """
        Build call rows for Neo4j from resolved edges.

        Args:
            edges: Dictionary of edges with aggregated data
            project_name: Project name

        Returns:
            List of safe call row dictionaries
        """
        call_rows: List[Dict[str, Any]] = []

        for (caller_id, callee_id, kind), agg in edges.items():
            try:
                safe_row = {
                    "project_name": str(project_name),
                    "caller_id": str(caller_id),
                    "callee_id": str(callee_id),
                    "kind": str(kind),
                    "count": int(agg.get("count") or 0),
                    "lines": [int(x) for x in (agg.get("lines") or []) if x is not None],
                }
                call_rows.append(safe_row)
            except Exception:
                # Skip malformed rows
                continue

        return call_rows

    @staticmethod
    def _name_low(x: str) -> str:
        """Normalize name to lowercase for case-insensitive comparison"""
        return (x or "").casefold()
