from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse
import re
import requests
from bs4 import BeautifulSoup

from services.supplier_adapters import ADAPTERS, PUBLIC_PRICE_SUPPLIERS, direct_search_results, supplier_name_for_url, canonicalize_url, slugify_product_name

DEFAULT_SUPPLIER_DOMAINS = [domain for adapter in ADAPTERS for domain in adapter.domains]
SUPPLIER_NAME_HINTS = {domain: adapter.name for adapter in ADAPTERS for domain in adapter.domains}


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    supplier_hint: str = ""


def supplier_hint_from_url(url: str) -> str:
    return supplier_name_for_url(url)


def build_cas_supplier_queries(cas_number: str, chemical_name: str | None = None) -> list[str]:
    cas = cas_number.strip()
    chem = (chemical_name or "").strip()
    price_first_sites = " OR ".join(PUBLIC_PRICE_SUPPLIERS[:8])
    base_terms = [
        # v8 price-first discovery: look for public price-table language before broad supplier noise.
        f'"{cas}" "Size" "Price" "Stock"',
        f'"{cas}" "Pack Size" "Price"',
        f'"{cas}" "USD" "In stock"',
        f'"{cas}" "Bulk Inquiry" "Price"',
        f'"{cas}" "catalog no" "price"',
        f'"{cas}" ({price_first_sites})',
        # Broader fallbacks.
        f'"{cas}" supplier price',
        f'"{cas}" catalog price',
        f'"{cas}" buy chemical',
        f'"{cas}" quote',
    ]
    if chem:
        base_terms.extend([
            f'"{cas}" "{chem}" "Size" "Price"',
            f'"{chem}" "{cas}" price',
            f'"{chem}" "{cas}" "pack size"',
        ])
    return base_terms


def direct_supplier_search_urls(cas_number: str, tier: str | None = None) -> list[SearchResult]:
    """v8 adapter registry seed URLs. Defaults to all sources; caller can request price_first first."""
    return direct_search_results(cas_number.strip(), tier=tier)


def serpapi_search(
    queries: Iterable[str],
    api_key: str,
    max_results_per_query: int = 8,
    timeout: int = 20,
) -> list[SearchResult]:
    if not api_key:
        return []
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    endpoint = "https://serpapi.com/search.json"
    for query in queries:
        params = {"engine": "google", "q": query, "api_key": api_key, "num": max_results_per_query}
        try:
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        for item in payload.get("organic_results", [])[:max_results_per_query]:
            url = item.get("link") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(SearchResult(
                title=item.get("title") or "Untitled search result",
                url=url,
                snippet=item.get("snippet") or "",
                source="serpapi",
                supplier_hint=supplier_hint_from_url(url),
            ))
    return results


def filter_likely_supplier_results(results: list[SearchResult]) -> list[SearchResult]:
    filtered: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        haystack = f"{result.title} {result.url} {result.snippet}".lower()
        if any(domain in haystack for domain in DEFAULT_SUPPLIER_DOMAINS):
            filtered.append(result)
            continue
        if any(term in haystack for term in ["supplier", "price", "quote", "buy", "catalog", "chemical", "cas"]):
            filtered.append(result)
    return filtered


_PRODUCT_HINT_RE = re.compile(
    r"(\.html(?:\?|$)|\.htm(?:\?|$)|/products?/|/product/|/compound/|/shop/compound/|/p/[A-Za-z0-9_-]+/|/p/|/item/|/sku/|/catalog/|/detail)",
    re.I,
)
_BAD_LINK_RE = re.compile(
    r"(privacy|terms|basket|cart(?:/|$)|login|signin|sign-in|register|contact|about|careers|linkedin|facebook|twitter|youtube|instagram|cookie|pdf|sds|coa|msds|orders?$|order-status|quick-order|promotions|sustainable|all-product-categories|clear-all-filters|clear\s*filters|my-account|wishlist|compare|newsletter|distributor|faq|support|download)",
    re.I,
)
_GENERIC_TEXT_RE = re.compile(r"(?i)^(view|details?|learn more|read more|search|home|products?|all products|menu|shop|buy now|add to cart|notify me|data sheet)$")
_STRONG_PRODUCT_PATH_RE = re.compile(
    r"(?i)(/products?/[a-z0-9][a-z0-9._-]{2,}|/compound/[a-z0-9][a-z0-9._-]{2,}|/p/[A-Z0-9][A-Z0-9._-]+/|/[a-z][a-z0-9-]{2,}\.html(?:\?|$)|/shop/compound/)"
)
_MCE_PRODUCT_RE = re.compile(r"(?i)^https?://(?:www\.)?medchemexpress\.com/[a-z0-9][a-z0-9-]+\.html")


def _same_domain(url_a: str, url_b: str) -> bool:
    try:
        a = urlparse(url_a).netloc.replace("www.", "")
        b = urlparse(url_b).netloc.replace("www.", "")
        return a and b and (a == b or a.endswith("." + b) or b.endswith("." + a))
    except Exception:
        return False


def _clean_short(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def _node_context(a_tag, limit: int = 1800) -> str:
    contexts = []
    for parent in [a_tag, a_tag.parent, a_tag.find_parent("li"), a_tag.find_parent("tr"), a_tag.find_parent("article"), a_tag.find_parent("section"), a_tag.find_parent("div")]:
        if parent is None:
            continue
        txt = parent.get_text(" ", strip=True)
        if txt and txt not in contexts:
            contexts.append(txt)
    return _clean_short(" | ".join(contexts), limit)


def _page_text_has_cas(page_text: str, cas_number: str) -> bool:
    return bool(cas_number and cas_number.lower() in (page_text or "").lower())


def _link_score(href: str, text: str, context: str, page_has_cas: bool, cas_number: str) -> int:
    """Score product links using card-level CAS evidence, not page-level CAS noise."""
    hay = f"{href} {text} {context}".lower()
    context_hay = f"{text} {context}".lower()
    text_clean = re.sub(r"\s+", " ", text or "").strip()
    score = 0
    href_product = bool(_STRONG_PRODUCT_PATH_RE.search(href) or _MCE_PRODUCT_RE.search(href))
    cas_l = cas_number.lower() if cas_number else ""
    cas_in_href = bool(cas_l and cas_l in href.lower())
    cas_in_card = bool(cas_l and cas_l in context_hay)
    cas_in_link_or_context = cas_in_href or cas_in_card

    if cas_in_link_or_context:
        score += 90
    if href_product:
        score += 35
    elif _PRODUCT_HINT_RE.search(hay):
        score += 18

    if page_has_cas and href_product and cas_in_link_or_context:
        score += 25
    elif page_has_cas and href_product:
        score += 4

    if text_clean and 3 <= len(text_clean) <= 120 and re.search(r"[A-Za-z]", text_clean) and not _GENERIC_TEXT_RE.search(text_clean):
        score += 12
    if any(term in hay for term in ["price", "pricing", "$", "pack", "size", "purity", "assay", "cas", "stock", "catalog no", "cat. no"]):
        score += 10
    if _BAD_LINK_RE.search(hay):
        score -= 140
    if len(text_clean) < 3 and not cas_in_link_or_context:
        score -= 25
    return score


def _source_label(result: SearchResult, page_has_cas: bool) -> str:
    if page_has_cas:
        return "expanded_product_link_v19_cas_page"
    return "expanded_product_link_v19"


def discover_product_links_from_page(result: SearchResult, cas_number: str, timeout: int = 12, max_links: int = 8) -> list[SearchResult]:
    """Open a supplier/search/CAS page and pull strong product-detail candidates.

    v19 fixes a key issue found in testing: many supplier CAS result pages link to
    a product slug that does not itself contain the CAS (e.g. MedChemExpress
    /phillyrin.html). If the source page contains the requested CAS, we lower the
    threshold for clean product-looking links while still excluding account/cart/nav links.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }
    cookies = {
        "currency": "USD",
        "Currency": "USD",
        "country": "US",
        "Country": "US",
        "countryCode": "US",
        "country_code": "US",
        "shipping_country": "US",
        "selected_country": "US",
        "locale": "en_US",
        "region": "US",
        "shipToCountry": "US",
    }
    try:
        resp = requests.get(result.url, headers=headers, cookies=cookies, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    page_has_cas = _page_text_has_cas(page_text, cas_number) or _page_text_has_cas(resp.url, cas_number)

    candidates: list[tuple[int, SearchResult]] = []
    seen: set[str] = set()

    # If a seed URL has already landed on a product-like page and mentions the CAS,
    # surface it as an expanded candidate so it is prioritized over fallback search pages.
    if page_has_cas and _STRONG_PRODUCT_PATH_RE.search(resp.url) and not _BAD_LINK_RE.search(resp.url):
        canon_self = canonicalize_url(resp.url)
        seen.add(canon_self)
        candidates.append((95, SearchResult(
            title=_clean_short(soup.title.get_text(" ", strip=True) if soup.title else result.title),
            url=resp.url,
            snippet=f"Seed landed on CAS-confirmed product-like page from {result.url}.",
            source="expanded_landing_product_v19",
            supplier_hint=result.supplier_hint or supplier_hint_from_url(resp.url),
        )))

    # v19: Some supplier CAS pages (notably MedChemExpress) expose the CAS and
    # compound title but do not make the price table visible until the clean product
    # slug page is opened. If the CAS page title looks like "Phillyrin 487-41-2",
    # synthesize the product URL directly instead of relying on an anchor card.
    if page_has_cas and 'medchemexpress.com' in resp.url.lower():
        title_text = _clean_short(soup.title.get_text(' ', strip=True) if soup.title else result.title, 160)
        slug_seed = re.sub(re.escape(cas_number), ' ', title_text, flags=re.I)
        slug_seed = re.split(r'\||-|–|—|MedChemExpress|MCE', slug_seed, maxsplit=1, flags=re.I)[0].strip()
        slug = slugify_product_name(slug_seed)
        if slug:
            product_url = f'https://www.medchemexpress.com/{slug}.html'
            canon_slug = canonicalize_url(product_url)
            if canon_slug not in seen:
                seen.add(canon_slug)
                candidates.append((120, SearchResult(
                    title=slug_seed or title_text,
                    url=product_url,
                    snippet=f'v19 MCE CAS-page product slug probe from {result.url}. Requested CAS {cas_number}. Page CAS match: {page_has_cas}.',
                    source='expanded_mce_cas_page_slug_probe_v19',
                    supplier_hint='MedChemExpress',
                )))

    for a in soup.find_all("a", href=True):
        href = urljoin(resp.url, a.get("href", ""))
        canon = canonicalize_url(href)
        if not href.startswith("http") or canon in seen:
            continue
        if not _same_domain(resp.url, href):
            continue
        text = _clean_short(a.get_text(" ", strip=True))
        context = _node_context(a)
        context_has_cas = bool(cas_number and cas_number.lower() in f"{text} {context} {href}".lower())
        score = _link_score(href, text, context, page_has_cas, cas_number)
        # v19: product slugs may omit CAS, but the surrounding result card must carry
        # the CAS unless the URL/title itself carries it. Otherwise unrelated products
        # on broad search pages leak into the evidence layer.
        threshold = 60 if context_has_cas else 92
        if score < threshold:
            continue
        seen.add(canon)
        candidates.append((score, SearchResult(
            title=text or result.title,
            url=href,
            snippet=f"Expanded from {result.url}. Page CAS match: {page_has_cas}. Link card CAS match: {context_has_cas}. Context: {context[:900]}",
            source=("expanded_product_link_v19_cas_context" if context_has_cas else "expanded_product_link_v19_needs_identity_check"),
            supplier_hint=result.supplier_hint or supplier_hint_from_url(result.url),
        )))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in candidates[:max_links]]
