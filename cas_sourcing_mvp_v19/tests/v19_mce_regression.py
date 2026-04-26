from bs4 import BeautifulSoup

from services.supplier_specific_parsers import supplier_specific_variant_rows
import services.page_extractor as pe


def test_mce_usd_price_ladder_parses_high_confidence():
    text = (
        "Phillyrin | Cytochrome P450 Activator CAS No. 487-41-2 Purity 98.82%. "
        "5 mg, USD 50, In-stock · Estimated Time of Arrival: April 20; "
        "10 mg, USD 80, In-stock · Estimated Time of Arrival: April 20; "
        "25 mg, USD 150, In-stock"
    )
    rows = supplier_specific_variant_rows("MedChemExpress", BeautifulSoup(text, "html.parser"), text)
    key = {(r["pack_size"], r["pack_unit"], r["price"], r["price_pairing_confidence"]) for r in rows}
    assert (5.0, "mg", 50.0, "HIGH") in key
    assert (10.0, "mg", 80.0, "HIGH") in key
    assert (25.0, "mg", 150.0, "HIGH") in key


def test_mce_fetch_failed_can_use_trusted_snippet_rows():
    original_fetch = pe._fetch
    try:
        def fail_fetch(*args, **kwargs):
            raise RuntimeError("simulated MCE block")
        pe._fetch = fail_fetch
        rows = pe.extract_product_rows_from_url(
            "487-41-2",
            "https://www.medchemexpress.com/phillyrin.html",
            supplier_hint="MedChemExpress",
            discovery_title="Phillyrin | Cytochrome P450 Activator",
            discovery_snippet=(
                "v19 MCE SerpAPI price-snippet probe for requested CAS 487-41-2. "
                "Phillyrin ; 5 mg, USD 50, In-stock ; 10 mg, USD 80, In-stock"
            ),
        )
    finally:
        pe._fetch = original_fetch
    assert len(rows) == 2
    assert all(r.cas_exact_match for r in rows)
    assert {(r.pack_size, r.pack_unit, r.listed_price_usd) for r in rows} == {(5.0, "mg", 50.0), (10.0, "mg", 80.0)}
    assert all(r.price_lead_type == "verified_public_price" for r in rows)


if __name__ == "__main__":
    test_mce_usd_price_ladder_parses_high_confidence()
    test_mce_fetch_failed_can_use_trusted_snippet_rows()
    print("v19 MCE regression tests passed")
