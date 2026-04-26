"""Small parser smoke tests for v19 regression guardrails.
Run with: python tests/v19_regression_smoke.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bs4 import BeautifulSoup
from services.supplier_specific_parsers import extract_supplier_specific_rows


def assert_ladder(supplier: str, text: str, expected_max: tuple[float, str, float]) -> None:
    rows, _, status = extract_supplier_specific_rows(supplier, BeautifulSoup("<html></html>", "html.parser"), text)
    mass = [r for r in rows if r["product_form"] == "solid/mass"]
    assert status == "supplier_specific_price_rows_found", status
    assert all(r["price_pairing_confidence"] == "HIGH" for r in mass), rows
    best = max(mass, key=lambda r: r["pack_size"])
    assert (best["pack_size"], best["pack_unit"], best["price"]) == expected_max, best


def main() -> None:
    assert_ladder(
        "TargetMol",
        "Phillyrin CAS No.: 487-41-2 Purity 99.71% Pack Size Price USA Stock Global Stock Quantity "
        "5 mg $39 In Stock 10 mg $64 In Stock 25 mg $129 In Stock 50 mg $188 In Stock "
        "100 mg $283 In Stock 500 mg $688 In Stock 1 mL x 10 mM (in DMSO) $46 In Stock",
        (500.0, "mg", 688.0),
    )
    assert_ladder(
        "ApexBio",
        "Forsythin CAS Number 487-41-2 Purity 98.51% Grouped product items Size Price Stock Qty "
        "1mL(10 mM in DMSO) $52.00 In stock 10mg $67.00 In stock 25mg $110.00 In stock "
        "50mg $155.00 In stock 100mg $217.00 In stock",
        (100.0, "mg", 217.0),
    )
    assert_ladder(
        "MedChemExpress",
        "Phillyrin CAS No: 487-41-2 5 mg, USD 50, In-stock 10 mg, USD 80, In-stock 50 mg, USD 275, In-stock",
        (50.0, "mg", 275.0),
    )
    assert_ladder(
        "Adooq",
        "Forsythin Product Information CAS Number 487-41-2 Grouped product items Size Price Stock Qty "
        "5mg $30.00 In stock 10mg $55.00 In stock 25mg $120.00 In stock 50mg $180.00 In stock "
        "100mg $275.00 In stock 500mg $680.00 In stock 10mM * 1mL in DMSO $40.00 In stock Free Delivery on orders over $500",
        (500.0, "mg", 680.0),
    )
    print("v19 parser regression smoke tests passed")


if __name__ == "__main__":
    main()
