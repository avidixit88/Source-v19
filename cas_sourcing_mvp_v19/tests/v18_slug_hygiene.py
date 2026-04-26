"""v19 regression guard: generated supplier probes must never contaminate other supplier URLs.
Run with: python tests/v19_slug_hygiene.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# supplier_product_url_candidates imports SearchResult from services.search_service.
# Stub it so this hygiene test stays lightweight and does not need requests/bs4.
@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    supplier_hint: str = ""

stub = types.ModuleType("services.search_service")
stub.SearchResult = SearchResult
sys.modules.setdefault("services.search_service", stub)

from services.supplier_adapters import (  # noqa: E402
    product_name_candidates_from_title,
    slugify_product_name,
    supplier_product_url_candidates,
)

BAD_URL_FRAGMENTS = (
    "apexbio-product-name-probe",
    "targetmol-product-name-probe",
    "medchemexpress-product-name-probe",
    "mce.html",
    "search?q=mce",
    "adooq.com/mce",
    "glpbio.com/mce",
    "biocrick.com/mce",
)


def assert_no_bad_urls(urls: list[str]) -> None:
    for url in urls:
        low = url.lower()
        assert not any(fragment in low for fragment in BAD_URL_FRAGMENTS), url


def main() -> None:
    assert product_name_candidates_from_title("ApexBio product-name probe: Phillyrin / Forsythin") == ["Phillyrin", "Forsythin"]
    assert slugify_product_name("MedChemExpress product-name probe: MCE") is None
    assert product_name_candidates_from_title("Adooq CAS search") == []

    contaminated = ["ApexBio product-name probe: Phillyrin / Forsythin", "MCE", "Adooq CAS search"]
    for supplier in ["AbMole", "Adooq", "BioCrick", "GLP Bio"]:
        urls = [r.url for r in supplier_product_url_candidates(supplier, contaminated, cas="487-41-2")]
        assert urls, supplier
        assert_no_bad_urls(urls)

    print("v19 slug hygiene regression tests passed")


if __name__ == "__main__":
    main()
