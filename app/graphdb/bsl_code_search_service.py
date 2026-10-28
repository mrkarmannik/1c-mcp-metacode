"""
BslCodeSearchService — single entry point for the search_bsl_code MCP tool.

Request pipeline:

    1. Cheap preflight (no embedding, no pinning):
           if BSL vector disabled OR SQLite vector_state stale/failed/missing
           OR vector_epoch != sqlite.current_epoch(scope) -> RLM path.
    2. Query embedding (only if preflight allowed vector path).
    3. Pin read_epoch = sqlite.current_epoch(scope). Revalidate vector_state.
       If state changed (Phase A of new reindex slipped in) -> RLM path.
    4. Vector path:
           Neo4j SEARCH ... VECTOR INDEX vec_bsl_code_unit ... LIMIT $top_k
           hybrid scorer over the candidate pool
               raw  -> vector_plus_field profile
               compr-> unified_top5_gated_margin0375 (confidence-gated base/tuned)
           load all units per parent (SQLite), apply 2-2-1 window selector
           cap to `limit` (total selected units, NOT routines)
    5. RLM path:
           body_bm25 top100 + intent routing rerank top50
           apply rlm_window_3600 or rlm_window_2200 depending on split strategy
           cap to `limit`.
    6. Group selected units by routine_id for the response shape.
       routine.score = max(unit.score for unit in routine_units).
       fragments ordered by line_start.

Output: list[BslCodeSearchResult] with stable fields routine_id, name,
signature, owner_qn, module_type, score, fragments / ranges.

No public "mode" parameter — internal path choice is logged but never surfaced.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config import settings
from neo4j.exceptions import Neo4jError

from . import bsl_code_rlm
from . import bsl_code_search_policy as search_policy
from . import embedding_usage_metrics as embedding_metrics
from .bsl_code_reranker import build_rerank_document
from .reranker import get_reranker
from .bsl_code_scorers import (
    bm25_score,
    expand_tokens_with_1c_synonyms,
    field_bm25_score,
    fuzzy_metadata_boost_score,
    generic_symbol_keys_set,
    idf_from_df,
    metadata_boost_score,
    quoted_phrase_tokens,
    quoted_symbol_boost_score,
    tokenize_1c_light,
)
from .bsl_code_selectors import (
    select_rlm_window_2200,
    select_rlm_window_3600,
    select_vector_window,
)
from .bsl_code_sqlite import (
    DOC_FIELD_KIND,
    FIELD_KINDS,
    BslCodeSqlite,
    UnitMeta,
    VectorState,
    get_bsl_code_sqlite,
)
from .cypher_templates import CYPHER_FETCH_ROUTINE_BODY_BATCH
from .embedding_text_format import (
    build_embedding_format_spec,
    compute_bsl_code_embedding_fingerprint,
    resolve_bsl_code_prompt_profile,
    resolve_effective_embedding_transport,
)

logger = logging.getLogger(__name__)


_bsl_rerank_warned_unavailable: bool = False


def _warn_bsl_rerank_unavailable_once() -> None:
    """One-shot misconfiguration warning: BSL rerank enabled but shared reranker unavailable."""
    global _bsl_rerank_warned_unavailable
    if _bsl_rerank_warned_unavailable:
        return
    _bsl_rerank_warned_unavailable = True
    logger.warning(
        "BSL rerank requested (BSL_CODE_RERANK_ENABLED=true) but shared reranker "
        "unavailable: RERANK_API_KEY missing or invalid"
    )


def reset_bsl_rerank_warning() -> None:
    """Test hook: reset the one-shot misconfig warning."""
    global _bsl_rerank_warned_unavailable
    _bsl_rerank_warned_unavailable = False


_METADATA_WEIGHT = 6
_METADATA_SYMBOL_WEIGHT = 1.0


class BslCodeSearchGraphUnavailable(RuntimeError):
    """Neo4j не ответил на чтение тел процедур (обрыв соединения, рестарт).

    Это не расхождение индекса с графом: переиндексация не запрашивается, клиенту возвращается
    просьба повторить запрос. В 2.0.1 такой сбой выглядел как «все процедуры пропали» и ставил
    reindex_requested — один поиск в первые секунды после рестарта Neo4j давал полную перестройку
    индекса (час и больше на крупной конфигурации).
    """


class BslCodeSearchIndexNotReady(RuntimeError):
    """Raised when search is requested before the first Phase A commit."""


@dataclass
class BslSearchUnitCandidate:
    unit_id: str
    routine_id: str
    unit_kind: str
    line_start: int
    line_end: int
    part_index: int
    part_total: int
    project_name: str
    config_name: str
    owner_qn: str
    owner_category: str
    module_type: str
    module_kind: str
    routine_type: str
    export: bool
    body_hash: str
    rel_path: str
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0


@dataclass
class BslCodeSearchResult:
    routine_id: str
    name: str
    signature: str
    owner_qn: str
    module_type: str
    score: float
    file_path: str = ""
    line: int = 0
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    ranges: List[Dict[str, int]] = field(default_factory=list)


@dataclass
class BslCodeSearchResponse:
    """Result wrapper that carries an optional notice for the MCP tool.

    `items` is the same shape as the legacy `search()` return value.
    `notice`, when present, is a dict describing edge-case behaviour the
    agent must know about (e.g. an explicit owner_categories mixed
    ordinary + excluded categories — only the ordinary part was searched).
    """
    items: List[BslCodeSearchResult] = field(default_factory=list)
    notice: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------- profiles


@dataclass(frozen=True)
class HybridProfile:
    name: str
    vector_weight: float
    lexical_weight: float
    field_weight: float
    metadata_boost_weight: float
    fuzzy_weight: float
    quoted_symbol_weight: float
    synonym_profile: str
    field_weights: Dict[str, float]
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    fuzzy_threshold: float = 0.88


_PROFILE_VECTOR_PLUS_FIELD = HybridProfile(
    name="vector_plus_field",
    vector_weight=0.65,
    lexical_weight=0.15,
    field_weight=0.20,
    metadata_boost_weight=0.02,
    fuzzy_weight=0.0,
    quoted_symbol_weight=0.0,
    synonym_profile="platform",
    field_weights={"symbol": 0.25, "object": 0.20, "path": 0.10, "body": 0.45},
)

_PROFILE_UNIFIED_TOP5_BASE = HybridProfile(
    name="unified_top5_base",
    vector_weight=0.5,
    lexical_weight=0.5,
    field_weight=0.1,
    metadata_boost_weight=0.02,
    fuzzy_weight=0.05,
    quoted_symbol_weight=0.05,
    synonym_profile="platform-ui",
    field_weights={"symbol": 0.25, "object": 0.20, "path": 0.10, "body": 0.45},
)

_PROFILE_UNIFIED_TOP5_TUNED = HybridProfile(
    name="unified_top5_tuned",
    vector_weight=0.45,
    lexical_weight=0.6,
    field_weight=0.1,
    metadata_boost_weight=0.06,
    fuzzy_weight=0.075,
    quoted_symbol_weight=0.1,
    synonym_profile="platform-ui",
    field_weights={"symbol": 0.25, "object": 0.20, "path": 0.10, "body": 0.45},
)

_GATE_MARGIN12 = 0.0375

# Compression strategies whose hybrid ranking uses the raw vector_plus_field
# profile instead of the gated compressed profiles. "none" plus the two
# raw-below-threshold strategies: in the latter short units stay raw, so the
# vector stays reliable and behaves like the raw index. Retrieval policy is
# owned here (search layer), not derived from bsl_code_compress.
_RAW_HYBRID_PROFILE_STRATEGIES = frozenset({
    "none",
    "rawbelow1000_lexdedup_terms_cap1_lines_normprefix",
    "rawbelow1000_lexdedup_cap1_nochainparts_lines_normprefix",
})


def _hybrid_weight_sum(profile: HybridProfile) -> float:
    return (
        profile.vector_weight
        + profile.lexical_weight
        + profile.field_weight
        + profile.metadata_boost_weight
        + profile.fuzzy_weight
        + profile.quoted_symbol_weight
    )


def _hybrid_weighted_average(
    profile: HybridProfile,
    *,
    vector: float,
    lexical: float,
    field: float,
    metadata: float,
    fuzzy: float,
    quoted: float,
) -> float:
    weighted_sum = (
        profile.vector_weight * vector
        + profile.lexical_weight * lexical
        + profile.field_weight * field
        + profile.metadata_boost_weight * metadata
        + profile.fuzzy_weight * fuzzy
        + profile.quoted_symbol_weight * quoted
    )
    weight_sum = _hybrid_weight_sum(profile)
    if weight_sum <= 0:
        return 0.0
    return weighted_sum / weight_sum


def _normalize_rlm_scores_by_top(candidates: List[BslSearchUnitCandidate]) -> None:
    if not candidates:
        return
    top_score = float(candidates[0].score)
    if top_score <= 1.0:
        return
    for candidate in candidates:
        candidate.score = float(candidate.score) / top_score


def compute_hard_ceiling() -> int:
    """
    Глобальный hard-потолок выдачи `search_bsl_code` = min(vector_top_k, 50).
    50 — приватная _RERANK_TOP_K в bsl_code_rlm.py (ceiling RLM пути).
    Используется и сервисом (top_k, normalization), и tool docstring builder'ом —
    единый источник правды, чтобы текст в описании инструмента не расходился
    с фактическим лимитом при изменении настройки bsl_code_vector_top_k.
    """
    return min(int(settings.bsl_code_vector_top_k or 50), 50)


def _normalize_excluded(
    raw: Optional[Sequence[str]],
    hard_ceiling: int,
) -> Optional[Tuple[str, ...]]:
    """
    Bounded normalization of cursor pagination state. Drops empty/falsy ids,
    deduplicates with first-occurrence order, truncates to `hard_ceiling`.
    Returns None on empty result so downstream branches in cypher/SQL/Python
    stay inactive.

    `hard_ceiling = min(bsl_code_vector_top_k, 50)` — единый общий лимит pipeline.
    `bsl_code_rerank_top_k` НЕ участвует: reranker имеет fallback пути
    (см. _apply_rerank_if_enabled), при которых пул возвращается без обрезания,
    так что rerank_top_k — это потолок только успешного rerank pass.
    """
    if not raw:
        return None
    seen: set = set()
    out: List[str] = []
    for uid in raw:
        if not uid:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
    if len(out) > hard_ceiling:
        out = out[:hard_ceiling]
    return tuple(out) if out else None


# --------------------------------------------------------------- search service


class BslCodeSearchService:
    def __init__(self, driver) -> None:
        self.driver = driver
        self.scope: str = settings.project_name
        self.sqlite: BslCodeSqlite = get_bsl_code_sqlite()

    def search(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        config_name: Optional[str] = None,
        owner_qn: Optional[str] = None,
        owner_qn_prefix: Optional[str] = None,
        owner_categories: Optional[Sequence[str]] = None,
        module_type: Optional[str] = None,
        routine_type: Optional[str] = None,
        export: Optional[bool] = None,
        include_fragments: bool = True,
        excluded_unit_ids: Optional[Sequence[str]] = None,
    ) -> List[BslCodeSearchResult]:
        """Backwards-compatible thin wrapper over search_with_notice — drops
        any notice. Internal callers that do not need the notice keep the
        same shape they had before the coverage-policy work."""
        return self.search_with_notice(
            query,
            limit=limit,
            config_name=config_name,
            owner_qn=owner_qn,
            owner_qn_prefix=owner_qn_prefix,
            owner_categories=owner_categories,
            module_type=module_type,
            routine_type=routine_type,
            export=export,
            include_fragments=include_fragments,
            excluded_unit_ids=excluded_unit_ids,
        ).items

    def search_with_notice(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        config_name: Optional[str] = None,
        owner_qn: Optional[str] = None,
        owner_qn_prefix: Optional[str] = None,
        owner_categories: Optional[Sequence[str]] = None,
        module_type: Optional[str] = None,
        routine_type: Optional[str] = None,
        export: Optional[bool] = None,
        include_fragments: bool = True,
        excluded_unit_ids: Optional[Sequence[str]] = None,
    ) -> BslCodeSearchResponse:
        if not query or not query.strip():
            return BslCodeSearchResponse(items=[], notice=None)

        # Per-request body cache (used by hybrid scorer and optional reranker
        # to avoid duplicate Routine.body fetches from Neo4j within one request).
        self._request_body_cache: Dict[str, Dict[str, Any]] = {}

        scope = self.scope
        effective_limit = max(1, int(limit or settings.bsl_code_search_default_limit))

        # Cursor pagination state (см. _normalize_excluded). При cursor_active
        # единый hard_ceiling прокидывается в _vector_path(top_k=...),
        # rlm_candidates(top_k=...) и normalize. При cursor=None non-cursor
        # pipeline идентичен текущему: vector top_k = settings.bsl_code_vector_top_k,
        # RLM использует приватный дефолт _RERANK_TOP_K=50.
        hard_ceiling = compute_hard_ceiling()
        normalized_excluded = _normalize_excluded(excluded_unit_ids, hard_ceiling)
        cursor_active = normalized_excluded is not None

        top_k = hard_ceiling if cursor_active else int(settings.bsl_code_vector_top_k)

        excluded_owner_categories = tuple(
            search_policy.normalize_excluded_categories(
                settings.bsl_code_embedding_excluded_owner_categories or ()
            )
        )
        exclude_regulated_reports = bool(
            settings.bsl_code_search_exclude_regulated_reports
        )

        # If the caller passed positive owner_categories, split into the part
        # that lives in default scope (included) and the part that matches
        # excluded policy (intersected). Mixed requests run only on the
        # included part and surface a notice telling the agent to issue a
        # second call for the excluded part.
        included, intersected = search_policy.split_owner_categories(
            owner_categories, excluded_owner_categories,
        )

        routing_notice: Optional[Dict[str, Any]] = None
        if owner_categories:
            if included and intersected:
                # mixed: search only included, surface notice
                effective_owner_categories: Optional[List[str]] = included
                routing_notice = {
                    "excluded_owner_categories_not_searched": list(intersected),
                    "message": (
                        "Some requested owner_categories overlap the configured "
                        "excluded-owner-categories policy. They were skipped in "
                        "this call. To search them, issue a separate call passing "
                        "ONLY the excluded categories in owner_categories."
                    ),
                }
            elif included:
                effective_owner_categories = included
            else:
                # caller asked ONLY for excluded categories — honour explicit
                # intent, skip vector path entirely (no vectors for them),
                # search RLM directly.
                effective_owner_categories = list(intersected)
        else:
            effective_owner_categories = None

        only_excluded_explicit = bool(
            owner_categories and intersected and not included
        )

        filters = _FilterSet(
            config_name=config_name,
            owner_qn=owner_qn,
            owner_qn_prefix=owner_qn_prefix,
            owner_categories=effective_owner_categories,
            module_type=module_type,
            routine_type=routine_type,
            export=export,
        )

        preflight_epoch = self.sqlite.current_epoch(scope)
        if preflight_epoch <= 0:
            logger.info("BSL search: scope=%s not indexed yet", scope)
            raise BslCodeSearchIndexNotReady(
                "BSL code search index is not ready yet. Indexing is in progress."
            )

        # Scoped reader-consistency gate: snapshot the state once per request so
        # the vector and RLM legs see the same `pending_routine_ids` /
        # `pending_rel_paths` view (cross-store between Neo4j and SQLite).
        try:
            scoped_state = self.sqlite.read_scoped_pending_state(scope)
        except Exception:
            scoped_state = {
                "scoped_apply_in_progress": False,
                "visibility_flip_done": False,
                "pending_routine_ids": set(),
                "pending_rel_paths": set(),
            }
        pending_routine_ids: Set[str] = set(
            scoped_state.get("pending_routine_ids") or ()
        )
        pending_rel_paths: Set[str] = set(
            scoped_state.get("pending_rel_paths") or ()
        )
        # While SQLite gate is on but Neo4j visibility flip hasn't completed,
        # vector results for affected routines are stale → skip vector path.
        conservative_path = bool(
            scoped_state.get("scoped_apply_in_progress")
            and not scoped_state.get("visibility_flip_done")
        )

        # Coverage policy for source-SQL legs.
        # Positive scope (explicit owner_categories) disables the negative
        # twin; regulated-reports flag is always honoured.
        # Vector path filters coverage via `code_embedding_visible` set by
        # the indexer, so vector cypher no longer receives these params.
        positive_active = bool(effective_owner_categories)
        rlm_excluded = (
            () if positive_active else excluded_owner_categories
        )

        selected_units: List[BslSearchUnitCandidate] = []
        read_epoch: Optional[int] = None

        if (
            not only_excluded_explicit
            and not conservative_path  # gate: vector may return stale until flip done
            and self._vector_eligible_preflight(scope, preflight_epoch)
        ):
            embedding_service = self._get_embedding_service_or_none()
            if embedding_service is None:
                logger.info("BSL search: embedding service unavailable, using RLM")
            else:
                query_vec = self._embed_query(query, embedding_service)
                if query_vec is None:
                    logger.info("BSL search: query embedding failed, using RLM")
                else:
                    pinned = self.sqlite.current_epoch(scope)
                    vec_state = self.sqlite.vector_state(scope)
                    if self._vector_eligible(vec_state, pinned):
                        vector_result = self._vector_path(
                            query, query_vec, pinned,
                            vec_state.vector_epoch or pinned,
                            filters, top_k, effective_limit,
                            excluded_unit_ids=normalized_excluded,
                            hard_ceiling=hard_ceiling,
                        )
                        if vector_result is None:
                            logger.info(
                                "BSL search: vector cypher failed for scope=%s, "
                                "falling back to RLM", scope,
                            )
                        else:
                            selected_units = vector_result
                            read_epoch = pinned
                    else:
                        logger.info(
                            "BSL search: vector became stale during embed "
                            "(read_epoch=%d, vector_epoch=%s, status=%s), using RLM",
                            pinned, vec_state.vector_epoch, vec_state.status,
                        )

        if read_epoch is None:
            read_epoch = self.sqlite.current_epoch(scope)
            selected_units = self._rlm_path(
                query, read_epoch, filters, effective_limit,
                excluded_owner_categories=rlm_excluded,
                exclude_regulated_reports=exclude_regulated_reports,
                excluded_unit_ids=normalized_excluded,
                rlm_top_k=hard_ceiling if cursor_active else None,
            )

        # Scoped reader gate (post-filter): drop any hit that belongs to a
        # routine or rel_path currently being mutated. With visibility_flip_done
        # the vector leg already excludes them via the Neo4j prefilter, but the
        # RLM leg reads SQLite which is being rewritten — post-filter here
        # guarantees both legs converge on the same view.
        if pending_routine_ids or pending_rel_paths:
            selected_units = [
                c for c in selected_units
                if (c.routine_id or "") not in pending_routine_ids
                and (c.rel_path or "") not in pending_rel_paths
            ]

        # Underfill compensation: under active scoped apply, vector pool can
        # shrink because invalidated routines no longer back the index, and
        # post-filter may further reduce. Top up from RLM (also gated) so the
        # user still gets `effective_limit` results when possible.
        if (
            scoped_state.get("scoped_apply_in_progress")
            and not conservative_path
            and read_epoch is not None
            and len(selected_units) < effective_limit
        ):
            rlm_fill = self._rlm_path(
                query, read_epoch, filters,
                effective_limit - len(selected_units) + max(3, effective_limit // 4),
                excluded_owner_categories=rlm_excluded,
                exclude_regulated_reports=exclude_regulated_reports,
                excluded_unit_ids=normalized_excluded,
                rlm_top_k=hard_ceiling if cursor_active else None,
            ) or []
            if pending_routine_ids or pending_rel_paths:
                rlm_fill = [
                    c for c in rlm_fill
                    if (c.routine_id or "") not in pending_routine_ids
                    and (c.rel_path or "") not in pending_rel_paths
                ]
            seen_uids = {c.unit_id for c in selected_units}
            seen_rids = {c.routine_id for c in selected_units}
            for c in rlm_fill:
                if c.unit_id in seen_uids or c.routine_id in seen_rids:
                    continue
                selected_units.append(c)
                seen_uids.add(c.unit_id)
                seen_rids.add(c.routine_id)
                if len(selected_units) >= effective_limit:
                    break
            selected_units = selected_units[:effective_limit]

        if not selected_units:
            return BslCodeSearchResponse(items=[], notice=routing_notice)

        items = self._materialize_results(
            selected_units, read_epoch, include_fragments, scope,
        )
        return BslCodeSearchResponse(items=items, notice=routing_notice)

    # ------------------------------------------------------------------ paths

    def _current_embedding_fingerprint(self) -> str:
        return compute_bsl_code_embedding_fingerprint(
            embedding_model=settings.embedding_model or "",
            embedding_prompt_mode=settings.bsl_code_embedding_prompt_mode or "auto",
            embedding_api_base=settings.embedding_api_base or "",
            embedding_transport_setting=getattr(
                settings, "embedding_transport", "auto",
            ) or "auto",
        )

    def _vector_eligible_preflight(self, scope: str, current_epoch: int) -> bool:
        if not settings.enable_bsl_code_search or not settings.enable_bsl_code_embedding:
            return False
        state = self.sqlite.vector_state(scope)
        if state.status != "ready" or state.vector_epoch is None:
            return False
        if state.vector_epoch != current_epoch:
            return False
        if state.embedding_fingerprint != self._current_embedding_fingerprint():
            return False
        return True

    def _vector_eligible(self, state: VectorState, read_epoch: int) -> bool:
        return (
            state.status == "ready"
            and state.vector_epoch is not None
            and state.vector_epoch == read_epoch
            and state.embedding_fingerprint == self._current_embedding_fingerprint()
        )

    def _embed_query(self, query: str, embedding_service) -> Optional[List[float]]:
        try:
            profile = resolve_bsl_code_prompt_profile(
                settings.embedding_model or "",
                settings.bsl_code_embedding_prompt_mode or "auto",
            )
        except ValueError as e:
            logger.warning("BSL search: invalid prompt mode: %s", e)
            return None
        transport = resolve_effective_embedding_transport(
            settings.embedding_api_base or "",
            getattr(settings, "embedding_transport", "auto") or "auto",
        )
        spec = build_embedding_format_spec(
            profile=profile,
            transport=transport,
            side="query",
            purpose="code",
            description_instruction="",
        )
        metric_started = embedding_metrics.started()
        try:
            batch_result = embedding_metrics.call_single_with_usage(
                embedding_service, query, format_spec=spec
            )
        except Exception as e:
            embedding_metrics.record_failure(
                event_type="bsl_code.embedding.query",
                embedding_service=embedding_service,
                duration_ms=embedding_metrics.elapsed_ms(metric_started),
            )
            logger.warning("BSL search: embedding API failed: %s", e)
            return None
        embedding_metrics.record_result(
            event_type="bsl_code.embedding.query",
            embedding_service=embedding_service,
            result=batch_result,
            duration_ms=embedding_metrics.elapsed_ms(metric_started),
        )
        embedding = embedding_metrics.first_embedding(batch_result)
        if embedding is None:
            return None
        return embedding

    def _vector_path(
        self,
        query_text: str,
        query_vec: List[float],
        read_epoch: int,
        vector_epoch: int,
        filters: "_FilterSet",
        top_k: int,
        effective_limit: int,
        excluded_unit_ids: Optional[Sequence[str]] = None,
        hard_ceiling: Optional[int] = None,
    ) -> Optional[List[BslSearchUnitCandidate]]:
        overfetch = 2 if filters.needs_post_filter() else 1
        categories: List[Optional[str]] = (
            list(filters.owner_categories)
            if filters.owner_categories
            else [None]
        )

        merged: Dict[str, Any] = {}
        try:
            with self.driver.session(database=settings.neo4j_database) as session:
                for cat in categories:
                    cypher, params = _build_vector_search_cypher(
                        self.scope, vector_epoch, filters, top_k,
                        overfetch=overfetch, owner_category=cat,
                        excluded_unit_ids=excluded_unit_ids,
                    )
                    params["query_vec"] = query_vec
                    for row in session.run(cypher, **params):
                        rec = dict(row)
                        uid = str(rec.get("unit_id") or "")
                        if not uid:
                            continue
                        prev = merged.get(uid)
                        if prev is None or float(rec.get("score") or 0.0) > float(prev.get("score") or 0.0):
                            merged[uid] = rec
        except Neo4jError as e:
            logger.warning("BSL search: vector SEARCH failed (%s); using RLM fallback", e)
            return None
        except Exception as e:
            logger.warning("BSL search: vector cypher failed (%s); using RLM fallback", e)
            return None

        rows = sorted(
            merged.values(),
            key=lambda r: float(r.get("score") or 0.0),
            reverse=True,
        )

        candidates: List[BslSearchUnitCandidate] = []
        for row in rows:
            cand = _row_to_candidate(row)
            if cand is None:
                continue
            if not filters.matches_post(cand):
                continue
            candidates.append(cand)

        if not candidates:
            return []

        # Augment with SQLite metadata for fields the Neo4j row didn't carry
        # (module_kind / rel_path live only in SQLite).
        metas = self.sqlite.fetch_unit_metadata(
            [c.unit_id for c in candidates], read_epoch
        )
        meta_by_id = {m.unit_id: m for m in metas}
        missing_meta_ids: List[str] = []
        for c in candidates:
            meta = meta_by_id.get(c.unit_id)
            if meta is None:
                missing_meta_ids.append(c.unit_id)
                continue
            c.line_start = meta.line_start
            c.line_end = meta.line_end
            c.part_index = meta.part_index
            c.part_total = meta.part_total
            c.body_hash = meta.body_hash or c.body_hash
            c.module_kind = meta.module_kind or c.module_kind
            c.rel_path = meta.rel_path or c.rel_path
        if missing_meta_ids:
            # Sidecar miss for vector candidates → drop them to keep file_path
            # honest in the public response. Same consistency contract as
            # `_materialize_results` body row / body_hash mismatch handlers.
            logger.warning(
                "BSL search: sidecar miss for vector candidates %s; requesting reindex",
                missing_meta_ids[:10],
            )
            self.sqlite.request_reindex(self.scope)
            drop = set(missing_meta_ids)
            candidates = [c for c in candidates if c.unit_id not in drop]
            if not candidates:
                return []

        # Hybrid scoring + selector.
        profile_choice = self._pick_hybrid_profile(candidates, query_text, read_epoch)
        self._apply_hybrid_blend(candidates, query_text, read_epoch, profile_choice)

        if excluded_unit_ids and hard_ceiling is not None:
            # F6 + F-S1: cap фактической глубины к hard_ceiling, только когда
            # cursor активен. Применяется после hybrid blend — глубинные
            # кандидаты, поднятые lexical/field/metadata сигналами, остаются
            # в top-hard_ceiling. Без cursor (non-cursor scoped) поведение
            # бит-в-бит идентично текущему (trim не активируется).
            candidates = sorted(
                candidates, key=lambda c: c.score, reverse=True,
            )[:hard_ceiling]

        candidates = self._apply_rerank_if_enabled(candidates, query_text, read_epoch)

        return self._apply_vector_window(
            candidates, read_epoch, effective_limit,
            excluded_unit_ids=excluded_unit_ids,
        )

    def _rlm_path(
        self,
        query: str,
        read_epoch: int,
        filters: "_FilterSet",
        effective_limit: int,
        excluded_owner_categories: Sequence[str] = (),
        exclude_regulated_reports: bool = False,
        excluded_unit_ids: Optional[Sequence[str]] = None,
        rlm_top_k: Optional[int] = None,
    ) -> List[BslSearchUnitCandidate]:
        # При cursor active передаём top_k=hard_ceiling, иначе rlm_candidates
        # использует приватный дефолт _RERANK_TOP_K=50 (текущее поведение).
        rlm_kwargs: Dict[str, Any] = dict(
            scope=self.scope,
            query=query,
            read_epoch=read_epoch,
            config_name=filters.config_name,
            module_type=filters.module_type,
            routine_type=filters.routine_type,
            export=filters.export,
            owner_categories=filters.owner_categories,
            excluded_owner_categories=excluded_owner_categories,
            exclude_regulated_reports=exclude_regulated_reports,
        )
        if rlm_top_k is not None:
            rlm_kwargs["top_k"] = int(rlm_top_k)
        raw = bsl_code_rlm.rlm_candidates(self.sqlite, **rlm_kwargs)

        # F1: cursor-state filter применяется ПОСЛЕ rlm_candidates (post-filter
        # из top-K пула), не в source SQL до LIMIT — иначе RLM уходил бы глубже
        # hard_ceiling. Симметрично vector path (там post-WITH WHERE).
        if excluded_unit_ids:
            excluded_set = set(excluded_unit_ids)
            raw = [
                (meta, score) for meta, score in raw
                if meta.unit_id not in excluded_set
            ]

        candidates: List[BslSearchUnitCandidate] = []
        for meta, score in raw:
            cand = _meta_to_candidate(meta, score, lexical=True)
            if not filters.matches_post(cand):
                continue
            candidates.append(cand)

        if not candidates:
            return []

        _normalize_rlm_scores_by_top(candidates)

        candidates = self._apply_rerank_if_enabled(candidates, query, read_epoch)

        selector = (
            select_rlm_window_3600
            if "3600" in (settings.bsl_code_split_strategy or "")
            else select_rlm_window_2200
        )
        return self._apply_window(
            candidates, read_epoch, effective_limit, selector,
            excluded_unit_ids=excluded_unit_ids,
        )

    # ------------------------------------------------------------------ rerank

    def _apply_rerank_if_enabled(
        self,
        candidates: List[BslSearchUnitCandidate],
        query_text: str,
        read_epoch: int,
    ) -> List[BslSearchUnitCandidate]:
        """
        Optional cross-encoder rerank pass.

        On success: returns reranked head (tail beyond BSL_CODE_RERANK_TOP_K is
        dropped) — selector receives a single-scale pool (Cohere relevance_score
        for all entries). On any failure or when reranker is disabled: returns
        the input list unchanged (current vector/RLM behaviour preserved).

        Body for the head is fetched via the shared per-request cache; for the
        vector path this is a no-op (hybrid already populated it), for the RLM
        path this is the first Neo4j roundtrip. When Neo4j body fetch yields
        nothing (Neo4j unreachable), rerank is silently skipped.
        """
        if len(candidates) < 2:
            return candidates
        if not bool(getattr(settings, "bsl_code_rerank_enabled", False)):
            return candidates
        reranker = get_reranker()
        if reranker is None:
            _warn_bsl_rerank_unavailable_once()
            return candidates

        top_k = max(1, int(getattr(settings, "bsl_code_rerank_top_k", 50) or 50))
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        head = sorted_cands[:top_k]

        body_by_unit = self._fetch_body_per_unit_for_candidates(head, read_epoch)
        if not body_by_unit:
            logger.warning(
                "BSL search: rerank skipped — no body text available "
                "(Neo4j unreachable or empty cache)"
            )
            return candidates

        body_cache = getattr(self, "_request_body_cache", {}) or {}
        field_texts_by_unit = self.sqlite.field_lookup(
            [c.unit_id for c in head], read_epoch,
        )
        documents: List[str] = []
        used: List[BslSearchUnitCandidate] = []
        for c in head:
            body_text = body_by_unit.get(c.unit_id)
            if not body_text:
                continue
            field_texts = field_texts_by_unit.get(c.unit_id, {})
            cached = body_cache.get(c.routine_id, {}) or {}
            routine_name = field_texts.get("symbol") or cached.get("name") or ""
            documents.append(build_rerank_document(
                metadata_type_ru=field_texts.get("metadata_type", ""),
                object_name=field_texts.get("object", ""),
                form_name=field_texts.get("form", ""),
                routine_type=c.routine_type,
                routine_name=routine_name,
                body_text=body_text,
            ))
            used.append(c)

        if len(documents) < 2:
            return candidates

        result = reranker.rerank(
            query_text, documents, top_n=len(documents),
            event_type="bsl_code.rerank",
        )
        if result is None:
            return candidates

        reranked: List[BslSearchUnitCandidate] = []
        for idx, score in result:
            if 0 <= idx < len(used):
                c = used[idx]
                c.score = float(score)
                reranked.append(c)
        if not reranked:
            return candidates

        logger.info(
            "BSL search: rerank applied — %d candidates reranked (from pool of %d)",
            len(reranked), len(candidates),
        )
        return reranked

    # ------------------------------------------------------------------ hybrid

    def _pick_hybrid_profile(
        self,
        candidates: List[BslSearchUnitCandidate],
        query_text: str,
        read_epoch: int,
    ) -> HybridProfile:
        """
        vector_plus_field for strategies in _RAW_HYBRID_PROFILE_STRATEGIES
        (none + rawbelow1000_* — short units stay raw, vector stays reliable).
        Fully-compressed strategies (lexdedup_*): confidence-gated choice between
        unified_top5_base (vector top1/top2 margin >= 0.0375) and
        unified_top5_tuned (low margin).
        """
        compression = (settings.bsl_code_compression_strategy or "none").strip().lower()
        if not compression or compression in _RAW_HYBRID_PROFILE_STRATEGIES:
            return _PROFILE_VECTOR_PLUS_FIELD

        if len(candidates) >= 2:
            margin = float(candidates[0].vector_score) - float(candidates[1].vector_score)
        else:
            margin = 0.0
        if margin >= _GATE_MARGIN12:
            return _PROFILE_UNIFIED_TOP5_BASE
        return _PROFILE_UNIFIED_TOP5_TUNED

    def _apply_hybrid_blend(
        self,
        candidates: List[BslSearchUnitCandidate],
        query_text: str,
        read_epoch: int,
        profile: HybridProfile,
    ) -> None:
        if not candidates:
            return

        unit_ids = [c.unit_id for c in candidates]

        # Load metadata fields (symbol, object, form, metadata_type, path)
        # from SQLite. field_lookup no longer returns 'body' — raw BSL
        # lives only in Neo4j Routine.body (plan invariant). Body for
        # the top-K candidates is fetched from Neo4j once here and
        # reused for field BM25 (body) + _doc reconstruction +
        # downstream fragment slicing (cached per request via
        # self._request_body_cache).
        field_texts = self.sqlite.field_lookup(unit_ids, read_epoch)

        # Batch-fetch Neo4j body once per request for all candidates.
        body_by_unit = self._fetch_body_per_unit_for_candidates(
            candidates, read_epoch,
        )

        # Build full field_tokens_by_id including the 'body' kind sourced
        # from the Neo4j slice — so field BM25 keeps working identically
        # to the old code, with field_lookup-driven body source replaced
        # by the per-request Neo4j fetch.
        field_tokens_by_id: Dict[str, Dict[str, List[str]]] = {}
        for uid in unit_ids:
            meta = field_texts.get(uid, {})
            by_kind: Dict[str, List[str]] = {
                fk: tokenize_1c_light(meta.get(fk, "")) for fk in FIELD_KINDS
            }
            by_kind["body"] = tokenize_1c_light(body_by_unit.get(uid, ""))
            field_tokens_by_id[uid] = by_kind

        # Reconstruct chunk_search_text tokens per candidate, body sourced
        # from the Neo4j slice instead of field_lookup.
        doc_tokens_by_id: Dict[str, List[str]] = {}
        for c in candidates:
            meta = field_texts.get(c.unit_id, {})
            doc_tokens_by_id[c.unit_id] = self._reconstruct_doc_tokens(
                c, meta, body_text=body_by_unit.get(c.unit_id, ""),
            )

        # Expand the query with the profile's synonym map.
        base_tokens = tokenize_1c_light(query_text)
        expanded_tokens = expand_tokens_with_1c_synonyms(
            base_tokens, weight=1, profile=profile.synonym_profile,
        )

        # IDF + avgdl for body and per-field BM25 (global, from Phase A).
        doc_count, avgdl_doc = self.sqlite.read_corpus_stats(self.scope, read_epoch, DOC_FIELD_KIND)
        idf_doc_df = self.sqlite.read_idf(self.scope, read_epoch, DOC_FIELD_KIND, set(expanded_tokens))
        idf_doc = idf_from_df(idf_doc_df, doc_count)

        field_idf: Dict[str, Tuple[Dict[str, float], float]] = {}
        for fk in FIELD_KINDS:
            fk_count, fk_avgdl = self.sqlite.read_corpus_stats(self.scope, read_epoch, fk)
            fk_df = self.sqlite.read_idf(self.scope, read_epoch, fk, set(expanded_tokens))
            field_idf[fk] = (idf_from_df(fk_df, fk_count), fk_avgdl)

        generic_keys = generic_symbol_keys_set() if profile.metadata_boost_weight > 0 else None
        quoted_phrases = quoted_phrase_tokens(query_text)

        # Raw component arrays (before normalization).
        vec_scores: List[float] = []
        lex_scores: List[float] = []
        field_scores: List[float] = []
        meta_scores: List[float] = []
        fuzzy_scores: List[float] = []
        quoted_scores: List[float] = []

        for c in candidates:
            doc_tokens = doc_tokens_by_id.get(c.unit_id, [])
            vec_scores.append(float(c.vector_score))
            lex_scores.append(bm25_score(
                expanded_tokens, doc_tokens, idf_doc, avgdl_doc or 1.0,
                profile.bm25_k1, profile.bm25_b,
            ))
            field_scores.append(field_bm25_score(
                expanded_tokens, c.unit_id, field_tokens_by_id, field_idf,
                profile.field_weights, profile.bm25_k1, profile.bm25_b,
            ))
            meta_scores.append(metadata_boost_score(
                expanded_tokens, c.unit_id, field_tokens_by_id,
                symbol_weight=1.0, object_weight=0.6, form_weight=0.4,
                generic_symbol_keys=generic_keys,
                generic_symbol_penalty=0.25,
            ))
            if profile.fuzzy_weight > 0:
                fuzzy_scores.append(fuzzy_metadata_boost_score(
                    expanded_tokens, c.unit_id, field_tokens_by_id,
                    symbol_weight=1.0, object_weight=0.8,
                    form_weight=0.4, metadata_type_weight=0.3,
                    threshold=profile.fuzzy_threshold,
                ))
            else:
                fuzzy_scores.append(0.0)
            if profile.quoted_symbol_weight > 0 and quoted_phrases:
                quoted_scores.append(quoted_symbol_boost_score(
                    quoted_phrases, c.unit_id, field_tokens_by_id,
                ))
            else:
                quoted_scores.append(0.0)

        vec_n = _normalize(vec_scores)
        lex_n = _normalize(lex_scores)
        field_n = _normalize(field_scores)
        meta_n = _normalize(meta_scores)
        fuzzy_n = _normalize(fuzzy_scores)
        quoted_n = _normalize(quoted_scores)

        for i, c in enumerate(candidates):
            score = _hybrid_weighted_average(
                profile,
                vector=vec_n[i],
                lexical=lex_n[i],
                field=field_n[i],
                metadata=meta_n[i],
                fuzzy=fuzzy_n[i],
                quoted=quoted_n[i],
            )
            c.lexical_score = float(lex_n[i])
            c.score = float(score)

    def _reconstruct_doc_tokens(
        self,
        c: BslSearchUnitCandidate,
        field_texts: Dict[str, str],
        body_text: str = "",
    ) -> List[str]:
        """
        Rebuild chunk_search_text tokens for a candidate. Body is no
        longer stored in field_lookup — caller passes the Neo4j-sourced
        slice via `body_text`. Mirrors bsl_code_indexer._build_search_text.
        """
        meta_parts = [
            field_texts.get("metadata_type", ""),
            field_texts.get("object", ""),
            field_texts.get("form", ""),
            c.module_kind,
            field_texts.get("path", ""),
        ]
        meta_segment = " ".join(p for p in meta_parts if p)
        symbol = field_texts.get("symbol", "")
        body = body_text or ""
        if _METADATA_WEIGHT <= 0:
            return tokenize_1c_light(body)
        symbol_repeats = round(_METADATA_WEIGHT * _METADATA_SYMBOL_WEIGHT)
        parts: List[str] = []
        if meta_segment:
            parts.extend([meta_segment] * _METADATA_WEIGHT)
        if symbol and symbol_repeats > 0:
            parts.extend([symbol] * symbol_repeats)
        parts.append(body)
        return tokenize_1c_light(" ".join(p for p in parts if p))

    def _fetch_body_per_unit_for_candidates(
        self,
        candidates: List[BslSearchUnitCandidate],
        read_epoch: int,
    ) -> Dict[str, str]:
        """
        Batch-fetch Routine.body from Neo4j for the top-K candidates and
        return a {unit_id: sliced_body} mapping. Body is sliced by each
        unit's line_start/line_end (from SQLite). Cached per request via
        self._request_body_cache so the downstream fragment slicing in
        the response builder doesn't issue a second roundtrip.
        """
        if not candidates:
            return {}
        # Read line ranges from SQLite for each candidate.
        unit_ids = [c.unit_id for c in candidates]
        meta_rows = self.sqlite.fetch_unit_metadata(unit_ids, read_epoch)
        by_uid: Dict[str, Any] = {m.unit_id: m for m in meta_rows}
        unique_routine_ids = sorted({
            by_uid[uid].routine_id for uid in unit_ids if uid in by_uid
        })
        if not unique_routine_ids:
            return {}
        # Skip Neo4j roundtrip for routines already cached in this request
        # (vector path populates the cache before rerank; rerank in RLM path
        # also pre-populates for response-builder reuse).
        cache = getattr(self, "_request_body_cache", None)
        if cache is None:
            self._request_body_cache = {}
            cache = self._request_body_cache
        missing = [rid for rid in unique_routine_ids if rid not in cache]
        if missing:
            fetched = self._fetch_bodies(missing) or {}
            cache.update(fetched)
        bodies = {rid: cache[rid] for rid in unique_routine_ids if rid in cache}
        # Slice each unit's body by exact char range (Drift 2 fix):
        # scorer body BM25 / _doc reconstruction must use the same byte
        # excerpt that Phase A indexed. Lines-based slicing diverged when
        # AST-safe split boundary fell mid-line. Fall back to line-based
        # slicing only for rows written before char ranges existed.
        out: Dict[str, str] = {}
        for uid in unit_ids:
            meta = by_uid.get(uid)
            if meta is None:
                continue
            body_info = bodies.get(meta.routine_id)
            if body_info is None:
                continue
            full_body = body_info.get("body") or ""
            char_start = getattr(meta, "char_start", 0) or 0
            char_end = getattr(meta, "char_end", 0) or 0
            if char_end > char_start:
                out[uid] = full_body[char_start:char_end]
            else:
                out[uid] = _slice_body_by_lines(
                    full_body, meta.line_start, meta.line_end,
                )
        return out

    # ------------------------------------------------------------------ windowing

    def _apply_vector_window(
        self,
        candidates: List[BslSearchUnitCandidate],
        read_epoch: int,
        effective_limit: int,
        excluded_unit_ids: Optional[Sequence[str]] = None,
    ) -> List[BslSearchUnitCandidate]:
        return self._apply_window(
            candidates, read_epoch, effective_limit, select_vector_window,
            excluded_unit_ids=excluded_unit_ids,
        )

    def _apply_window(
        self,
        candidates: List[BslSearchUnitCandidate],
        read_epoch: int,
        effective_limit: int,
        selector,
        excluded_unit_ids: Optional[Sequence[str]] = None,
    ) -> List[BslSearchUnitCandidate]:
        # Sort candidates by score so selector sees scored input in the same
        # order as the reference (which sorts parts by score before passing).
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        parts = [_candidate_to_part(c) for c in sorted_candidates]

        # Load all units per parent so the selector can expand windows beyond
        # the vector top-K.
        # F3: при cursor active фильтруем all_by_parent против excluded — иначе
        # selector через _meta_from_part_lookup мог бы вернуть excluded sibling
        # того же routine со score=0.0. parts уже отфильтрованы upstream
        # (vector cypher post-WITH WHERE / _rlm_path post-filter), но
        # all_by_parent живёт независимым путём через sqlite.all_units_by_parent.
        excluded_set = set(excluded_unit_ids) if excluded_unit_ids else None
        parent_ids = {c.routine_id for c in sorted_candidates}
        all_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for rid in parent_ids:
            metas = self.sqlite.all_units_by_parent(rid, read_epoch)
            parts_for_rid = [_meta_to_part(m) for m in metas]
            if excluded_set:
                parts_for_rid = [
                    p for p in parts_for_rid
                    if p["part_id"] not in excluded_set
                ]
            all_by_parent[rid] = parts_for_rid

        selected_parts = selector(parts, all_by_parent, effective_limit)

        # Map back to BslSearchUnitCandidate, preserving any score and parent
        # info from the original pool where available; for window-expanded
        # units that were not in the candidate pool, build a candidate from
        # the SQLite meta with score=0.0 (they ride on their parent's score).
        by_id = {c.unit_id: c for c in sorted_candidates}
        result: List[BslSearchUnitCandidate] = []
        seen: set = set()
        for part in selected_parts:
            uid = part["part_id"]
            if uid in seen:
                continue
            seen.add(uid)
            existing = by_id.get(uid)
            if existing is not None:
                result.append(existing)
                continue
            meta = _meta_from_part_lookup(part, all_by_parent)
            if meta is not None:
                result.append(_meta_to_candidate(meta, 0.0, lexical=False))
        return result[:effective_limit]

    # ------------------------------------------------------------------ response

    def _materialize_results(
        self,
        selected_units: List[BslSearchUnitCandidate],
        read_epoch: int,
        include_fragments: bool,
        scope: str,
    ) -> List[BslCodeSearchResult]:
        # Group by routine_id while preserving the selection order.
        groups: Dict[str, List[BslSearchUnitCandidate]] = {}
        for u in selected_units:
            groups.setdefault(u.routine_id, []).append(u)

        # Batch fetch routine bodies/signatures.
        routine_ids = list(groups.keys())
        body_map = self._fetch_bodies(routine_ids)
        if body_map is None:
            # Сбой чтения из Neo4j — не повод для переиндексации (BslCodeSearchGraphUnavailable).
            raise BslCodeSearchGraphUnavailable(
                "Neo4j не ответил при чтении тел процедур; повторите запрос через несколько секунд"
            )

        # Order routines by best unit score; within a routine, order fragments
        # by line_start.
        ordered_ids = sorted(
            groups.keys(),
            key=lambda rid: max(u.score for u in groups[rid]),
            reverse=True,
        )

        results: List[BslCodeSearchResult] = []
        for rid in ordered_ids:
            row = body_map.get(rid)
            if row is None:
                logger.warning(
                    "BSL search: routine_id=%s is in the index but absent in Neo4j; requesting reindex", rid,
                )
                self.sqlite.request_reindex(scope)
                continue
            body = row.get("body") or ""
            body_hash_neo = row.get("body_hash") or ""
            routine_line = int(row.get("line") or 0)
            units = sorted(groups[rid], key=lambda u: u.line_start)
            best = max(units, key=lambda u: u.score)

            if best.body_hash and body_hash_neo and best.body_hash != body_hash_neo:
                logger.warning(
                    "BSL search: body_hash mismatch for routine_id=%s "
                    "(sqlite=%s.., neo4j=%s..); dropping unit, requesting reindex",
                    rid, best.body_hash[:12], body_hash_neo[:12],
                )
                self.sqlite.request_reindex(scope)
                continue

            result = BslCodeSearchResult(
                routine_id=rid,
                name=row.get("name") or "",
                signature=row.get("signature") or "",
                owner_qn=best.owner_qn,
                module_type=best.module_type,
                score=round(float(best.score), 4),
                file_path=best.rel_path or "",
                line=routine_line,
            )
            for u in units:
                # u.line_start/u.line_end are body-relative (slice of Routine.body).
                # Public start_line/end_line are file-absolute via Routine.line.
                # routine_line == 0 → corrupted/legacy row, signal 0/0 honestly.
                file_start = routine_line + u.line_start - 1 if routine_line else 0
                file_end = routine_line + u.line_end - 1 if routine_line else 0
                if include_fragments:
                    code = _slice_body_by_lines(body, u.line_start, u.line_end)
                    result.fragments.append({
                        "fragment_id": u.unit_id,
                        "start_line": file_start,
                        "end_line": file_end,
                        "code": code,
                    })
                else:
                    result.ranges.append({
                        "fragment_id": u.unit_id,
                        "start_line": file_start,
                        "end_line": file_end,
                    })
            results.append(result)
        return results

    def _fetch_bodies(self, routine_ids: List[str]) -> "Dict[str, Dict[str, Any]] | None":
        """Тела и сигнатуры процедур из Neo4j. None — чтение не удалось (сбой Neo4j); пустой
        словарь — таких процедур в графе нет. В 2.0.1 оба случая были пустым словарём."""
        if not routine_ids:
            return {}
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            with self.driver.session(database=settings.neo4j_database) as session:
                res = session.run(
                    CYPHER_FETCH_ROUTINE_BODY_BATCH,
                    routine_ids=routine_ids,
                )
                for record in res:
                    rid = record["routine_id"]
                    rows[rid] = {
                        "body": record["body"],
                        "body_hash": record["body_hash"],
                    }
            with self.driver.session(database=settings.neo4j_database) as session:
                res = session.run(
                    "UNWIND $rids AS rid MATCH (r:Routine {id: rid}) "
                    "RETURN r.id AS routine_id, r.name AS name, r.signature AS signature, "
                    "coalesce(r.line, 0) AS line",
                    rids=routine_ids,
                )
                for record in res:
                    rid = record["routine_id"]
                    if rid in rows:
                        rows[rid]["name"] = record["name"] or ""
                        rows[rid]["signature"] = record["signature"] or ""
                        rows[rid]["line"] = int(record["line"] or 0)
        except Exception as e:
            logger.warning("BSL search: failed to fetch routine bodies: %s", e)
            return None
        return rows

    def _get_embedding_service_or_none(self):
        try:
            from .embedding_service import get_embedding_service
            return get_embedding_service()
        except Exception as e:
            logger.debug("BSL search: get_embedding_service failed: %s", e)
            return None


# ============================================================ helpers / structs


@dataclass
class _FilterSet:
    config_name: Optional[str] = None
    owner_qn: Optional[str] = None
    owner_qn_prefix: Optional[str] = None
    owner_categories: Optional[List[str]] = None
    module_type: Optional[str] = None
    routine_type: Optional[str] = None
    export: Optional[bool] = None

    def needs_post_filter(self) -> bool:
        return bool(self.owner_qn or self.owner_qn_prefix)

    def matches_post(self, c: BslSearchUnitCandidate) -> bool:
        if self.owner_qn and c.owner_qn != self.owner_qn:
            return False
        if self.owner_qn_prefix and not c.owner_qn.startswith(self.owner_qn_prefix):
            return False
        return True


def _row_to_candidate(row) -> Optional[BslSearchUnitCandidate]:
    try:
        node_kind = row["unit_kind"]
        return BslSearchUnitCandidate(
            unit_id=row["unit_id"],
            routine_id=row["routine_id"],
            unit_kind=node_kind,
            line_start=int(row["line_start"] or 0),
            line_end=int(row["line_end"] or 0),
            part_index=int(row["part_index"] or 0),
            part_total=int(row["part_total"] or 1),
            project_name=row["project_name"] or "",
            config_name=row["config_name"] or "",
            owner_qn=row["owner_qn"] or "",
            owner_category=row["owner_category"] or "",
            module_type=row["module_type"] or "",
            module_kind="",
            routine_type=row["routine_type"] or "",
            export=bool(row["export"]),
            body_hash=row["body_hash"] or "",
            rel_path="",
            score=float(row["score"] or 0.0),
            vector_score=float(row["score"] or 0.0),
        )
    except Exception as e:
        logger.debug("BSL search: malformed cypher row (%s): %s", e, dict(row))
        return None


def _meta_to_candidate(meta: UnitMeta, score: float, *, lexical: bool) -> BslSearchUnitCandidate:
    return BslSearchUnitCandidate(
        unit_id=meta.unit_id,
        routine_id=meta.routine_id,
        unit_kind=meta.unit_kind,
        line_start=meta.line_start,
        line_end=meta.line_end,
        part_index=meta.part_index,
        part_total=meta.part_total,
        project_name=meta.project_name,
        config_name=meta.config_name,
        owner_qn=meta.owner_qn,
        owner_category=meta.owner_category,
        module_type=meta.module_type,
        module_kind=meta.module_kind,
        routine_type=meta.routine_type,
        export=meta.export,
        body_hash=meta.body_hash,
        rel_path=meta.rel_path,
        score=float(score),
        lexical_score=float(score) if lexical else 0.0,
    )


def _candidate_to_part(c: BslSearchUnitCandidate) -> Dict[str, Any]:
    return {
        "part_id": c.unit_id,
        "parent_id": c.routine_id,
        "part_index": c.part_index,
        "score": float(c.score),
        "line_start": c.line_start,
        "line_end": c.line_end,
    }


def _meta_to_part(m: UnitMeta) -> Dict[str, Any]:
    return {
        "part_id": m.unit_id,
        "parent_id": m.routine_id,
        "part_index": m.part_index,
        "score": 0.0,
        "line_start": m.line_start,
        "line_end": m.line_end,
        "_meta": m,
    }


def _meta_from_part_lookup(
    part: Dict[str, Any],
    all_by_parent: Dict[str, List[Dict[str, Any]]],
) -> Optional[UnitMeta]:
    if "_meta" in part:
        return part["_meta"]
    siblings = all_by_parent.get(part.get("parent_id", "")) or []
    for sibling in siblings:
        if sibling.get("part_id") == part.get("part_id"):
            return sibling.get("_meta")
    return None


def _slice_body_by_lines(body: str, line_start: int, line_end: int) -> str:
    if not body:
        return ""
    lines = body.splitlines(keepends=True)
    if not lines:
        return body
    s = max(1, int(line_start)) - 1
    e = max(s, int(line_end))
    return "".join(lines[s:e]).rstrip("\n")


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _build_vector_search_cypher(
    project_name: str,
    vector_epoch: int,
    filters: _FilterSet,
    top_k: int,
    overfetch: int,
    owner_category: Optional[str] = None,
    excluded_unit_ids: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    effective_k = max(1, int(top_k * max(1, overfetch)))

    # Coverage policy (excluded categories + regulated reports) is encoded
    # on the node as `code_embedding_visible`, recomputed by the indexer
    # whenever policy changes. Restricted vector `SEARCH WHERE` does not
    # accept `NOT … IN …` or `coalesce(...)`, so a single boolean
    # predicate is the only viable shape here.
    search_filters = [
        "n.project_name = $project_name",
        "n.code_embedding_epoch = $vector_epoch",
        "n.code_embedding_visible = true",
    ]
    params: Dict[str, Any] = {
        "project_name": project_name,
        "vector_epoch": int(vector_epoch),
        "top_k": effective_k,
    }
    if filters.config_name:
        search_filters.append("n.config_name = $config_name")
        params["config_name"] = filters.config_name
    if filters.module_type:
        search_filters.append("n.module_type = $module_type")
        params["module_type"] = filters.module_type
    if filters.routine_type:
        search_filters.append("n.routine_type = $routine_type")
        params["routine_type"] = filters.routine_type
    if filters.export is not None:
        search_filters.append("n.export = $export")
        params["export"] = bool(filters.export)
    if owner_category:
        search_filters.append("n.owner_category = $owner_category")
        params["owner_category"] = owner_category

    search_where = "\n      AND ".join(search_filters)

    # Cursor pagination filter (excluded_unit_ids) уходит в post-WITH WHERE,
    # не в SEARCH WHERE: restricted vector SEARCH WHERE не принимает NOT IN
    # (см. комментарий выше).
    excluded_predicate = ""
    if excluded_unit_ids:
        excluded_predicate = "\n    WHERE NOT (n.id IN $excluded_unit_ids)"
        params["excluded_unit_ids"] = list(excluded_unit_ids)

    cypher = f"""
    MATCH (n:BslCodeSearchUnit)
      SEARCH n IN (
        VECTOR INDEX vec_bsl_code_unit
        FOR $query_vec
        WHERE {search_where}
        LIMIT $top_k
      ) SCORE AS score
    WITH n, score,
         CASE WHEN 'RoutineCodeUnit' IN labels(n) THEN 'routine_code_unit' ELSE 'routine' END AS unit_kind{excluded_predicate}
    RETURN
        n.id AS unit_id,
        CASE WHEN unit_kind = 'routine' THEN n.id ELSE n.routine_id END AS routine_id,
        unit_kind,
        coalesce(n.line_start, 1) AS line_start,
        coalesce(n.line_end, 1)   AS line_end,
        coalesce(n.part_index, 0) AS part_index,
        coalesce(n.part_total, 1) AS part_total,
        n.project_name   AS project_name,
        n.config_name    AS config_name,
        n.owner_qn       AS owner_qn,
        n.owner_category AS owner_category,
        n.module_type    AS module_type,
        n.routine_type   AS routine_type,
        coalesce(n.export, false) AS export,
        coalesce(n.body_hash, '') AS body_hash,
        score
    ORDER BY score DESC
    """
    return cypher, params
