from __future__ import annotations

from collections import defaultdict
from urllib.parse import urlparse
import pandas as pd

from services.search_service import (
    build_cas_supplier_queries,
    direct_supplier_search_urls,
    filter_likely_supplier_results,
    serpapi_search,
    discover_product_links_from_page,
    SearchResult,
)
from services.page_extractor import extract_product_rows_from_url
from core.procurement_logic import enrich_procurement_trust
from services.supplier_adapters import (
    ADAPTERS,
    canonicalize_url,
    extract_snippet_price,
    classify_price_visibility,
    best_action_for_status,
    supplier_key_from_url,
    product_name_candidates_from_title,
    supplier_product_url_candidates,
    slugify_product_name,
)
from services.supplier_specific_parsers import parser_name_for_supplier


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen = set()
    out = []
    for result in results:
        key = canonicalize_url(result.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(result)
    return out


def _supplier_key(result: SearchResult) -> str:
    return result.supplier_hint or supplier_key_from_url(result.url)


def _clean_pack(row: pd.Series) -> str:
    size = row.get("pack_size")
    unit = row.get("pack_unit")
    if pd.isna(size) or not unit or pd.isna(unit):
        return ""
    try:
        return f"{float(size):g} {unit}"
    except Exception:
        return f"{size} {unit}"

def _pack_size_to_g_value(size, unit) -> float | None:
    try:
        val = float(size)
    except Exception:
        return None
    if not pd.notna(val) or val <= 0:
        return None
    u = str(unit or "").strip().lower()
    factors = {"ug": 1e-6, "µg": 1e-6, "μg": 1e-6, "mcg": 1e-6, "mg": 1e-3, "g": 1.0, "kg": 1000.0}
    factor = factors.get(u)
    return val * factor if factor is not None else None


def _collapse_price_status(statuses: list[str]) -> str:
    priority = [
        "Public price extracted",
        "Search-snippet price only",
        "Login/account price required",
        "Quote required",
        "No public price detected by current parser",
        "No public price detected",
        "Extraction failed",
    ]
    for status in priority:
        if status in statuses:
            return status
    return statuses[0] if statuses else "No public price detected by current parser"


def _valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _looks_like_search_or_account_url(url: str) -> bool:
    hay = str(url or "").lower()
    noisy_markers = [
        "/search", "catalogsearch", "keyword=", "search=", "q=", "query=",
        "login", "signin", "register", "cart", "basket", "order-status", "quick-order"
    ]
    return any(marker in hay for marker in noisy_markers)


def _choose_representative_url(urls: list[str]) -> str:
    """Prefer a durable product-detail URL over search/session URLs for the UI Open Source link."""
    clean = [u for u in dict.fromkeys(str(x).strip() for x in urls if str(x).strip()) if _valid_http_url(u)]
    if not clean:
        return ""
    for u in clean:
        if not _looks_like_search_or_account_url(u):
            return u
    return clean[0]




def _mce_serpapi_price_snippet_results(cas_number: str, product_names: set[str], serpapi_key: str | None) -> list[SearchResult]:
    """Use optional SerpAPI to recover MCE public price snippets for confirmed names.

    This is intentionally scoped to MedChemExpress because MCE product pages often
    expose rows like "5 mg, USD 50, In-stock" in public search snippets while the
    page may reject plain server requests. The snippet is only generated after a
    CAS-confirmed product name has been discovered from another supplier/source.
    """
    if not serpapi_key or not product_names:
        return []
    queries = []
    for name in sorted(product_names)[:4]:
        slug = slugify_product_name(name)
        if not slug:
            continue
        queries.append(f'site:medchemexpress.com/{slug}.html "{name}" "USD" "In-stock"')
        queries.append(f'site:medchemexpress.com "{cas_number}" "{name}" "USD"')
    if not queries:
        return []
    results = serpapi_search(queries[:6], serpapi_key, max_results_per_query=3)
    out: list[SearchResult] = []
    seen: set[str] = set()
    for r in results:
        if "medchemexpress.com" not in r.url.lower():
            continue
        canon = canonicalize_url(r.url)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(SearchResult(
            title=r.title,
            url=r.url,
            snippet=f"v19 MCE SerpAPI price-snippet probe for requested CAS {cas_number}. {r.snippet}",
            source="v19_mce_serpapi_price_snippet_probe",
            supplier_hint="MedChemExpress",
        ))
    return out

def _safe_extract_products(cas_number: str, result: SearchResult, supplier: str):
    """Never let one bad supplier page crash the whole Streamlit run."""
    try:
        return extract_product_rows_from_url(
            cas_number,
            result.url,
            supplier_hint=supplier,
            discovery_title=result.title,
            discovery_snippet=result.snippet,
        )
    except Exception as exc:
        from services.page_extractor import ExtractedProductData
        return [ExtractedProductData(
            supplier=supplier or supplier_key_from_url(result.url),
            title=result.title or "Extraction failed",
            cas_exact_match=False,
            purity=None,
            pack_size=None,
            pack_unit=None,
            listed_price_usd=None,
            stock_status="Not visible",
            product_url=result.url,
            extraction_status=f"failed: {type(exc).__name__}: {str(exc)[:180]}",
            confidence=0,
            evidence=f"v19 guarded extraction failure; source preserved for manual review: {str(exc)[:300]}",
            extraction_method="guarded_exception",
            raw_matches="",
            catalog_number=None,
            price_visibility_status="Extraction failed",
            best_action="Open source manually",
            adapter_name=supplier,
            price_pairing_confidence="NONE",
            supplier_parser_name=parser_name_for_supplier(supplier),
            supplier_parser_status="guarded_exception_before_supplier_parser",
            identity_reason="guarded exception before identity check",
            observed_cas_numbers="",
            price_lead_type="none",
        )]


def summarize_supplier_rows(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df.copy()
    df = detail_df.copy()
    df["pack_label"] = df.apply(_clean_pack, axis=1)
    records = []
    for supplier, g in df.groupby("supplier", dropna=False):
        visible = g[g.get("listed_price_usd").notna()] if "listed_price_usd" in g.columns else pd.DataFrame()
        eligible = g[g.get("bulk_estimate_eligible", pd.Series(False, index=g.index)).fillna(False).astype(bool)] if "bulk_estimate_eligible" in g.columns else pd.DataFrame()
        verified = g[g.get("verified_public_price", pd.Series(False, index=g.index)).fillna(False).astype(bool)] if "verified_public_price" in g.columns else visible[visible.get("cas_exact_match", pd.Series(False, index=visible.index)).fillna(False).astype(bool)] if not visible.empty else pd.DataFrame()
        anchor_source = eligible if not eligible.empty else verified if not verified.empty else visible
        largest_pack_label = "Not visible"
        largest_pack_price = None
        lowest_unit_price = None
        if not anchor_source.empty:
            mass_anchor = anchor_source.copy()
            if "pack_size_g" not in mass_anchor.columns or mass_anchor["pack_size_g"].isna().all():
                mass_anchor["pack_size_g"] = mass_anchor.apply(lambda r: _pack_size_to_g_value(r.get("pack_size"), r.get("pack_unit")), axis=1)
            mass_anchor = mass_anchor[mass_anchor["pack_size_g"].notna()].copy()
            if not mass_anchor.empty:
                anchor_row = mass_anchor.sort_values(["pack_size_g", "listed_price_usd"], ascending=[False, True]).iloc[0]
                largest_pack_label = _clean_pack(anchor_row)
                largest_pack_price = anchor_row.get("listed_price_usd")
                if "price_per_g" in mass_anchor.columns and mass_anchor["price_per_g"].notna().any():
                    lowest_unit_price = float(mass_anchor["price_per_g"].min())
                elif "listed_price_usd" in mass_anchor.columns:
                    tmp = mass_anchor[mass_anchor["listed_price_usd"].notna()].copy()
                    if not tmp.empty:
                        lowest_unit_price = float((tmp["listed_price_usd"] / tmp["pack_size_g"]).min())
        statuses = [str(x) for x in g.get("price_visibility_status", pd.Series(dtype=str)).dropna().tolist()]
        status = _collapse_price_status(statuses)
        pack_options = sorted({x for x in g["pack_label"].tolist() if x})
        purities = sorted({str(x) for x in g.get("purity", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "Not visible"})
        urls = list(dict.fromkeys((g.get("product_url", pd.Series(dtype=str)).dropna().astype(str).tolist() + g.get("landing_url", pd.Series(dtype=str)).dropna().astype(str).tolist() + g.get("canonical_product_url", pd.Series(dtype=str)).dropna().astype(str).tolist())))[:10]
        cat_nums = sorted({str(x) for x in g.get("catalog_number", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"})
        source_tiers = sorted({str(x) for x in g.get("source_tier", pd.Series(dtype=str)).dropna().tolist() if str(x)})
        product_forms = sorted({str(x) for x in g.get("product_form", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"})
        trust_decisions = list(dict.fromkeys(g.get("procurement_trust_decision", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]
        observed_cas = sorted({str(x) for x in g.get("observed_cas_numbers", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"})
        identity_reasons = list(dict.fromkeys(g.get("identity_reason", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]
        parser_names = sorted({str(x) for x in g.get("supplier_parser_name", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"})
        parser_statuses = list(dict.fromkeys(g.get("supplier_parser_status", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]
        row = {
            "supplier": supplier,
            "cas_number": g["cas_number"].iloc[0],
            "cas_exact_match": bool(g.get("cas_exact_match", pd.Series([False])).fillna(False).astype(bool).any()),
            "source_tier": ", ".join(source_tiers) if source_tiers else "unknown",
            "products_found": int(g["canonical_url"].nunique()),
            "catalog_numbers": ", ".join(cat_nums[:8]) if cat_nums else "Not extracted",
            "purities_found": ", ".join(purities[:8]) if purities else "Not visible",
            "pack_options": ", ".join(pack_options[:12]) if pack_options else "Not visible",
            "product_forms": ", ".join(product_forms) if product_forms else "unknown",
            "bulk_estimate_eligible_count": int(len(eligible)),
            "visible_price_count": int(len(visible)),
            "verified_public_price_count": int(g.get("verified_public_price", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "verified_public_price" in g.columns else int(visible.get("cas_exact_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not visible.empty else 0,
            "candidate_price_count": int(g.get("tentative_price", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "tentative_price" in g.columns else int((~visible.get("cas_exact_match", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not visible.empty else 0,
            "cas_confirmed_visible_price_count": int(visible.get("cas_exact_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not visible.empty else 0,
            "cas_unconfirmed_visible_price_count": int((~visible.get("cas_exact_match", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not visible.empty else 0,
            "high_confidence_price_pairs": int((visible.get("price_pairing_confidence", pd.Series(dtype=str)) == "HIGH").sum()) if not visible.empty else 0,
            "medium_confidence_price_pairs": int((visible.get("price_pairing_confidence", pd.Series(dtype=str)) == "MEDIUM").sum()) if not visible.empty else 0,
            "low_confidence_price_pairs": int((visible.get("price_pairing_confidence", pd.Series(dtype=str)) == "LOW").sum()) if not visible.empty else 0,
            "best_visible_price_usd": float(visible["listed_price_usd"].min()) if not visible.empty else None,
            "largest_verified_pack": largest_pack_label,
            "largest_verified_pack_price_usd": float(largest_pack_price) if pd.notna(largest_pack_price) else None,
            "lowest_verified_unit_price_per_g": float(lowest_unit_price) if lowest_unit_price is not None else None,
            "price_visibility_status": status,
            "best_action": best_action_for_status(status),
            "stock_summary": "; ".join(list(dict.fromkeys(g.get("stock_status", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]) or "Not visible",
            "max_extraction_confidence": int(g.get("extraction_confidence", pd.Series([0])).fillna(0).max()),
            "source_urls": " | ".join(urls),
            "representative_url": _choose_representative_url(urls),
            "trust_decisions": " | ".join(trust_decisions) if trust_decisions else "",
            "observed_cas_numbers": " | ".join(observed_cas[:6]) if observed_cas else "",
            "identity_reasons": " | ".join(identity_reasons) if identity_reasons else "",
            "supplier_parser_names": ", ".join(parser_names) if parser_names else parser_name_for_supplier(str(supplier)),
            "supplier_parser_statuses": " | ".join(parser_statuses) if parser_statuses else "not_checked",
            "notes": "v19 grouped row. CAS-confirmed public prices drive the model; lower-confidence candidates remain visible. Largest verified pack is surfaced separately so the scale-up anchor is auditable.",
            "data_source": "live_supplier_adapter_summary_v19",
        }
        records.append(row)
    out = pd.DataFrame(records)
    if not out.empty:
        out["_has_public_price"] = out["visible_price_count"] > 0
        out["_tier_rank"] = out["source_tier"].map(lambda x: 3 if "price_first" in str(x) else 2 if "marketplace" in str(x) else 1)
        out = out.sort_values(
            ["cas_exact_match", "_has_public_price", "_tier_rank", "max_extraction_confidence", "products_found"],
            ascending=[False, False, False, False, False],
        ).drop(columns=["_has_public_price", "_tier_rank"])
    return out


def _build_supplier_seed_map(cas_number: str, serpapi_key: str | None, chemical_name: str | None) -> tuple[dict[str, list[SearchResult]], pd.DataFrame]:
    seed_map: dict[str, list[SearchResult]] = defaultdict(list)
    discovery_records = []

    # 1) Registry-first: every curated supplier gets a chance before generic search can dominate.
    direct = direct_supplier_search_urls(cas_number)
    for result in direct:
        seed_map[_supplier_key(result)].append(result)

    # 2) Optional search API broadens coverage, but it is subordinate to the registry and identity-gated later.
    serp_results: list[SearchResult] = []
    if serpapi_key:
        queries = build_cas_supplier_queries(cas_number, chemical_name)
        serp_results = filter_likely_supplier_results(serpapi_search(queries, serpapi_key or ""))
        for result in serp_results:
            seed_map[_supplier_key(result)].append(result)

    for supplier, results in seed_map.items():
        for r in _dedupe_results(results):
            discovery_records.append({
                "supplier": supplier,
                "title": r.title,
                "url": r.url,
                "canonical_url": canonicalize_url(r.url),
                "domain": _domain(r.url),
                "snippet": r.snippet,
                "source": r.source,
                "supplier_hint": r.supplier_hint,
            })
    return {k: _dedupe_results(v) for k, v in seed_map.items()}, pd.DataFrame(discovery_records)




def build_supplier_coverage_report(discovery_df: pd.DataFrame, detail_df: pd.DataFrame, max_suppliers: int, max_pages_to_extract: int) -> pd.DataFrame:
    """Supplier-by-supplier audit table for the curated registry.

    This is deliberately separate from product evidence. It tells us whether each source was
    walked, which parser profile ran, and whether missing prices are likely account-gated,
    quote-only, skipped by budget, or simply not parsed by the current profile.
    """
    adapter_by_name = {a.name: a for a in ADAPTERS}
    walked = set()
    if discovery_df is not None and not discovery_df.empty and "supplier" in discovery_df.columns:
        walked = set(discovery_df["supplier"].dropna().astype(str).tolist()[:max_suppliers])
    records = []
    for idx, adapter in enumerate(ADAPTERS, start=1):
        supplier = adapter.name
        dseed = discovery_df[discovery_df["supplier"].astype(str).eq(supplier)] if discovery_df is not None and not discovery_df.empty and "supplier" in discovery_df.columns else pd.DataFrame()
        drows = detail_df[detail_df["supplier"].astype(str).eq(supplier)] if detail_df is not None and not detail_df.empty and "supplier" in detail_df.columns else pd.DataFrame()
        row_count = int(len(drows))
        cas_rows = int(drows.get("cas_exact_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if row_count else 0
        public_mask = drows.get("listed_price_usd", pd.Series(dtype=float)).notna() if row_count else pd.Series(dtype=bool)
        cas_mask = drows.get("cas_exact_match", pd.Series(dtype=bool)).fillna(False).astype(bool) if row_count else pd.Series(dtype=bool)
        public_rows = int(public_mask.sum()) if row_count else 0
        cas_confirmed_public_rows = int((public_mask & cas_mask).sum()) if row_count else 0
        cas_unconfirmed_public_rows = int((public_mask & ~cas_mask).sum()) if row_count else 0
        high_rows = int((drows.get("price_pairing_confidence", pd.Series(dtype=str)).astype(str) == "HIGH").sum()) if row_count else 0
        med_rows = int((drows.get("price_pairing_confidence", pd.Series(dtype=str)).astype(str) == "MEDIUM").sum()) if row_count else 0
        low_rows = int((drows.get("price_pairing_confidence", pd.Series(dtype=str)).astype(str) == "LOW").sum()) if row_count else 0
        parser_names = sorted({str(x) for x in drows.get("supplier_parser_name", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"}) if row_count else [parser_name_for_supplier(supplier)]
        parser_statuses = list(dict.fromkeys(drows.get("supplier_parser_status", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5] if row_count else []
        statuses = [str(x) for x in drows.get("price_visibility_status", pd.Series(dtype=str)).dropna().tolist()] if row_count else []
        fetch_failed_rows = int(drows.get("extraction_method", pd.Series(dtype=str)).astype(str).str.contains("fetch_failed|guarded_exception", case=False, na=False).sum()) if row_count else 0
        walked_flag = idx <= max_suppliers
        if idx > max_suppliers:
            coverage_status = "skipped_by_supplier_limit"
        elif cas_confirmed_public_rows > 0:
            coverage_status = "cas_confirmed_public_price_parsed"
        elif cas_unconfirmed_public_rows > 0:
            coverage_status = "public_price_candidate_cas_unconfirmed"
        elif any("fetch_failed" in s.lower() or "guarded_exception" in s.lower() for s in parser_statuses):
            coverage_status = "fetch_failed_or_blocked"
        elif any("location_prompt" in s.lower() for s in parser_statuses):
            coverage_status = "location_prompt_us_default_attempted"
        elif any("js_price_loader" in s.lower() for s in parser_statuses):
            coverage_status = "js_price_loader_or_dynamic_price"
        elif any("Login" in s for s in statuses) or any("login" in s.lower() for s in parser_statuses):
            coverage_status = "checked_login_or_account_price"
        elif any("Quote" in s for s in statuses) or any("quote" in s.lower() or "inquiry" in s.lower() for s in parser_statuses):
            coverage_status = "checked_quote_or_rfq"
        elif row_count > 0 and cas_rows > 0:
            coverage_status = "cas_confirmed_no_public_price_parsed"
        elif walked_flag:
            coverage_status = "walked_no_product_rows_accepted"
        else:
            coverage_status = "not_checked_or_no_rows"
        records.append({
            "registry_order": idx,
            "supplier": supplier,
            "source_tier": adapter.source_tier,
            "expected_pricing_behavior": adapter.expected_behavior,
            "public_price_likelihood": adapter.public_price_likelihood,
            "walked_by_current_settings": bool(idx <= max_suppliers),
            "seed_urls": int(len(dseed)),
            "product_evidence_rows": row_count,
            "cas_confirmed_rows": cas_rows,
            "public_price_rows": public_rows,
            "cas_confirmed_public_price_rows": cas_confirmed_public_rows,
            "cas_unconfirmed_public_price_rows": cas_unconfirmed_public_rows,
            "fetch_failed_rows": fetch_failed_rows,
            "high_confidence_price_rows": high_rows,
            "medium_confidence_price_rows": med_rows,
            "low_confidence_price_rows": low_rows,
            "supplier_parser_names": ", ".join(parser_names),
            "supplier_parser_statuses": " | ".join(parser_statuses) if parser_statuses else "not_checked_or_no_rows",
            "coverage_status": coverage_status,
            "notes": "v19 coverage: CAS-confirmed public prices are model-eligible; CAS-unconfirmed public prices are shown as lower-confidence evidence; fetch failures usually mean site blocking, wrong search endpoint, region/location selection, session, or JS constraints.",
        })
    return pd.DataFrame(records)


def _append_extracted_row(extracted_rows: list[dict], extracted, result: SearchResult, supplier: str, adapter, cas_number: str, chemical_name: str | None, snippet_price: float | None, price_visibility_status: str) -> None:
    extracted_rows.append({
        "cas_number": cas_number, "chemical_name": chemical_name or "", "supplier": extracted.supplier,
        "source_tier": adapter.source_tier if adapter else "unknown", "expected_pricing_behavior": adapter.expected_behavior if adapter else "unknown",
        "region": "Unknown", "purity": extracted.purity or "Not visible", "pack_size": extracted.pack_size, "pack_unit": extracted.pack_unit,
        "listed_price_usd": extracted.listed_price_usd, "snippet_price_usd": snippet_price if extracted.cas_exact_match else None,
        "price_visibility_status": price_visibility_status, "best_action": best_action_for_status(price_visibility_status), "stock_status": extracted.stock_status,
        "lead_time": "Not visible", "product_url": extracted.product_url, "canonical_url": canonicalize_url(extracted.product_url), "domain": _domain(extracted.product_url),
        "catalog_number": extracted.catalog_number, "notes": extracted.evidence, "page_title": extracted.title, "cas_exact_match": extracted.cas_exact_match,
        "identity_reason": getattr(extracted, "identity_reason", ""), "observed_cas_numbers": getattr(extracted, "observed_cas_numbers", ""),
        "price_lead_type": getattr(extracted, "price_lead_type", "none"), "extraction_status": extracted.extraction_status, "extraction_confidence": extracted.confidence,
        "extraction_method": extracted.extraction_method, "price_pairing_confidence": getattr(extracted, "price_pairing_confidence", "NONE"), "raw_matches": extracted.raw_matches,
        "product_form": getattr(extracted, "product_form", "unknown"), "purity_confidence": getattr(extracted, "purity_confidence", "NONE"),
        "url_role": getattr(extracted, "url_role", "source_page"), "landing_url": getattr(extracted, "landing_url", extracted.product_url),
        "canonical_product_url": getattr(extracted, "canonical_product_url", canonicalize_url(extracted.product_url)), "price_noise_flag": getattr(extracted, "price_noise_flag", False),
        "supplier_parser_name": getattr(extracted, "supplier_parser_name", parser_name_for_supplier(supplier)), "supplier_parser_status": getattr(extracted, "supplier_parser_status", "not_checked"),
        "data_source": "live_extraction_v19_mce_adapter",
    })

def discover_live_suppliers(
    cas_number: str,
    chemical_name: str | None = None,
    serpapi_key: str | None = None,
    max_pages_to_extract: int = 96,
    include_direct_links: bool = True,
    max_suppliers: int = 42,
    pages_per_supplier: int = 3,
    required_purity: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """v19 strict supplier-registry sourcing with supplier-specific parser coverage.

    Main correction from v8: the engine now iterates suppliers from the curated registry first,
    gives each supplier its own extraction budget, and requires page-level CAS identity. This prevents
    one or two noisy vendors/search pages from consuming the extraction budget or falsely confirming
    wrong product pages.
    """
    seed_map, discovery_df = _build_supplier_seed_map(cas_number, serpapi_key, chemical_name)
    if not include_direct_links and not serpapi_key:
        seed_map = {}

    # Preserve registry priority order.
    supplier_order = [a.name for a in ADAPTERS if a.name in seed_map]
    supplier_order += [s for s in seed_map if s not in supplier_order]
    supplier_order = supplier_order[:max_suppliers]

    extracted_rows = []
    total_extracted = 0
    # v19: as soon as any supplier confirms the CAS/product name, use that name to
    # probe suppliers whose native CAS search pages are weak or JS-heavy. This is how
    # we reach pages like MedChemExpress /phillyrin.html or Ambeed /products/phillyrin.html.
    product_name_pool: set[str] = set()
    if chemical_name and chemical_name.strip() and chemical_name.strip().lower() not in {"acetaminophen", "optional"}:
        for name in product_name_candidates_from_title(chemical_name.strip()):
            product_name_pool.add(name)
    extra_discovery_records: list[dict] = []

    adapter_by_name = {a.name: a for a in ADAPTERS}
    processed_candidate_keys: set[tuple[str, str]] = set()

    for supplier in supplier_order:
        if total_extracted >= max_pages_to_extract:
            break
        seeds = seed_map.get(supplier, [])
        adapter = adapter_by_name.get(supplier)
        # Expand each supplier's own seed/search pages into product candidates.
        expanded: list[SearchResult] = []
        for seed in seeds[:3]:
            try:
                new_links = discover_product_links_from_page(seed, cas_number, max_links=8)
                expanded.extend(new_links)
                for link_result in new_links:
                    for name in product_name_candidates_from_title(link_result.title):
                        product_name_pool.add(name)
            except Exception:
                # Keep the seed page as fallback; link expansion is helpful but not allowed to crash discovery.
                continue
        # v19 supplier-specific product-name probes. These are especially important for
        # suppliers whose CAS page is a landing page and pricing lives one click deeper.
        product_probes = supplier_product_url_candidates(supplier, sorted(product_name_pool), cas=cas_number)
        # v19: MCE-specific optional search-snippet probes. These do not replace
        # product-page parsing; they recover public MCE pack/price rows when the
        # product page blocks plain fetches but public snippets expose the ladder.
        if supplier == "MedChemExpress":
            product_probes.extend(_mce_serpapi_price_snippet_results(cas_number, product_name_pool, serpapi_key))
        for probe in product_probes:
            extra_discovery_records.append({
                "supplier": supplier,
                "title": probe.title,
                "url": probe.url,
                "canonical_url": canonicalize_url(probe.url),
                "domain": _domain(probe.url),
                "snippet": probe.snippet,
                "source": probe.source,
                "supplier_hint": probe.supplier_hint,
            })
        candidates = _dedupe_results(expanded + product_probes + seeds)
        # Product-link candidates first, then product-name probes, then direct search pages as fallback.
        candidates = sorted(candidates, key=lambda r: (0 if (r.source.startswith("expanded") or r.source.startswith("v19_product_name_probe") or r.source.startswith("v19_product_name_probe") or r.source.startswith("v19_mce_serpapi_price_snippet_probe") or r.source.startswith("v19_deferred_product_name_probe")) else 1, canonicalize_url(r.url)))

        supplier_pages_done = 0
        for result in candidates:
            if total_extracted >= max_pages_to_extract or supplier_pages_done >= pages_per_supplier:
                break
            candidate_key = (supplier, canonicalize_url(result.url))
            if candidate_key in processed_candidate_keys:
                continue
            processed_candidate_keys.add(candidate_key)
            extracted_products = _safe_extract_products(cas_number, result, supplier)
            snippet_price = extract_snippet_price(result.snippet)

            page_kept = False
            for extracted in extracted_products:
                price_visibility_status = extracted.price_visibility_status
                if extracted.listed_price_usd is None and snippet_price is not None and extracted.cas_exact_match:
                    price_visibility_status = classify_price_visibility(None, result.snippet, snippet_price, extracted.extraction_status)

                # v19 keep rule: keep CAS-confirmed pages, meaningful supplier availability state,
                # and CAS-unconfirmed public price candidates as lower-confidence evidence.
                keep = bool(extracted.cas_exact_match) or price_visibility_status in [
                    "Public price extracted", "Search-snippet price only", "Login/account price required", "Quote required", "Extraction failed"
                ]
                if not keep:
                    continue

                _append_extracted_row(extracted_rows, extracted, result, supplier, adapter, cas_number, chemical_name, snippet_price, price_visibility_status)
                if extracted.cas_exact_match:
                    for name in product_name_candidates_from_title(extracted.title):
                        product_name_pool.add(name)
                    for name in product_name_candidates_from_title(result.title):
                        product_name_pool.add(name)
                page_kept = True
            if page_kept:
                supplier_pages_done += 1
                total_extracted += 1

    # v19 late product-name probe pass: revisit earlier suppliers after another source
    # reveals the compound name/slug, instead of trusting search-page prices.
    if product_name_pool and total_extracted < max_pages_to_extract:
        for supplier in supplier_order:
            if total_extracted >= max_pages_to_extract:
                break
            adapter = adapter_by_name.get(supplier)
            probes = _dedupe_results(supplier_product_url_candidates(supplier, sorted(product_name_pool), cas=cas_number))
            supplier_probe_done = 0
            for probe in probes[:2]:
                if total_extracted >= max_pages_to_extract or supplier_probe_done >= 1:
                    break
                candidate_key = (supplier, canonicalize_url(probe.url))
                if candidate_key in processed_candidate_keys:
                    continue
                processed_candidate_keys.add(candidate_key)
                extra_discovery_records.append({"supplier": supplier, "title": probe.title, "url": probe.url, "canonical_url": canonicalize_url(probe.url), "domain": _domain(probe.url), "snippet": probe.snippet + " Late v19 confirmed-name probe.", "source": "v19_late_confirmed_name_probe", "supplier_hint": supplier})
                extracted_products = _safe_extract_products(cas_number, probe, supplier)
                snippet_price = extract_snippet_price(probe.snippet)
                page_kept = False
                for extracted in extracted_products:
                    price_visibility_status = extracted.price_visibility_status
                    if extracted.listed_price_usd is None and snippet_price is not None and extracted.cas_exact_match:
                        price_visibility_status = classify_price_visibility(None, probe.snippet, snippet_price, extracted.extraction_status)
                    keep = bool(extracted.cas_exact_match) or (extracted.listed_price_usd is not None and not str(getattr(extracted, "price_lead_type", "")).startswith("search_page")) or price_visibility_status in ["Login/account price required", "Quote required", "Extraction failed", "Search page only; product page not resolved"]
                    if not keep:
                        continue
                    _append_extracted_row(extracted_rows, extracted, probe, supplier, adapter, cas_number, chemical_name, snippet_price, price_visibility_status)
                    if extracted.cas_exact_match:
                        for name in product_name_candidates_from_title(extracted.title):
                            product_name_pool.add(name)
                    page_kept = True
                if page_kept:
                    supplier_probe_done += 1
                    total_extracted += 1


    # v19: older deferred-probe block removed; late confirmed-name probes above now handle this
    # with identity-locked row appending and duplicate protection.

    if extra_discovery_records:
        discovery_df = pd.concat([discovery_df, pd.DataFrame(extra_discovery_records)], ignore_index=True).drop_duplicates(subset=["supplier", "canonical_url"], keep="first")

    detail_df = pd.DataFrame(extracted_rows)
    if not detail_df.empty:
        dedupe_cols = [c for c in ["supplier", "canonical_url", "catalog_number", "purity", "pack_size", "pack_unit", "listed_price_usd", "price_visibility_status", "price_pairing_confidence"] if c in detail_df.columns]
        detail_df = detail_df.drop_duplicates(subset=dedupe_cols, keep="first")
        # v19: keep CAS-unconfirmed public price rows as lower-confidence evidence; procurement_trust prevents them from entering the quantity model.
        detail_df = enrich_procurement_trust(detail_df, required_purity=required_purity)
        detail_df = detail_df.sort_values(["cas_exact_match", "source_tier", "bulk_estimate_eligible", "listed_price_usd", "extraction_confidence"], ascending=[False, True, False, True, False])

    summary_df = summarize_supplier_rows(detail_df)
    coverage_df = build_supplier_coverage_report(discovery_df, detail_df, max_suppliers=max_suppliers, max_pages_to_extract=max_pages_to_extract)
    return detail_df, discovery_df, summary_df, coverage_df
