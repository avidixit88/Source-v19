from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from bs4 import BeautifulSoup
from services.supplier_adapters import ADAPTERS

PRICE_RE = re.compile(
    r"(?:USD|US\$|\$)\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?|[0-9]+(?:\.[0-9]{1,4})?)|"
    r"\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?|[0-9]+(?:\.[0-9]{1,4})?)\s*(?:USD|US\s?dollars)\b",
    re.I,
)
PACK_RE = re.compile(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)\s?(ug|µg|μg|mcg|microgram|micrograms|mg|milligram|milligrams|g|gram|grams|kg|kilogram|kilograms|ml|mL|milliliter|milliliters|L|l|liter|liters)\b", re.I)
SOLUTION_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s?mL\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mM\s*(?:\(?\s*in\s*DMSO\s*\)?)?|\d+(?:\.\d+)?\s?mM\s*(?:x|\*)?\s*(\d+(?:\.\d+)?)\s?mL\s*(?:in\s*DMSO)?")
STOCK_RE = re.compile(r"(?i)(in\s*stock|out\s*of\s*stock|available|backorder|preorder|ships?\s*in\s*[^.;,|]{1,45}|usually\s*ships[^.;,|]{0,45}|lead\s*time[^.;,|]{0,45}|\d+[- ]?\d+\s*(?:days|weeks)|request\s+a?\s*quote|price\s+on\s+request)")
PRICE_NOISE_RE = re.compile(r"(?i)(free\s+shipping|orders?\s+over|minimum\s+order|cart\b|basket\b|subtotal|checkout|coupon|promo|discount|shipping\s+threshold|handling\s+fee|tax\b|recently\s+added|save\s+\d+%|reward|points)")
PACK_CONTEXT_NOISE_RE = re.compile(r"(?i)(molecular\s+weight|formula\s+weight|mw\b|g/mol|mol\s*wt|exact\s+mass)")
LOCATION_PROMPT_RE = re.compile(r"(?i)(choose\s+your\s+location|select\s+your\s+location|select\s+region|select\s+region\s+or\s+location|select\s+country|choose\s+country|ship\s+to|shipping\s+country|country/region|americas\s+.*united\s+states)")
JSON_PRICE_KEY_RE = re.compile(r"(?i)(price|finalprice|final_price|amount|unitprice|unit_price|listprice|list_price|saleprice|sale_price|yourprice|your_price|regularprice|regular_price)")
JSON_PACK_KEY_RE = re.compile(r"(?i)(size|pack|package|quantity|qty|amount|unit|name|label|sku|variant|option)")

# Suppliers where a visible ladder header such as "Pack Size / Price / Stock" is
# considered a structured table even when the website flattens it into text.
# This prevents regressions where TargetMol/ApexBio rows drop from HIGH to MEDIUM
# only because BeautifulSoup receives a text-rendered table rather than <tr>/<td>.
STRUCTURED_LADDER_SUPPLIERS = {
    "TargetMol", "ApexBio", "MedChemExpress", "Adooq", "SelleckChem",
    "GLP Bio", "AbMole", "Cayman Chemical", "ChemFaces", "CSNpharm",
}

@dataclass(frozen=True)
class SupplierParserProfile:
    supplier: str
    row_selectors: tuple[str, ...]
    option_selectors: tuple[str, ...]
    table_markers: tuple[str, ...]
    size_headers: tuple[str, ...]
    price_headers: tuple[str, ...]
    stock_headers: tuple[str, ...]
    product_path_hints: tuple[str, ...]
    parser_notes: str


def _default_profile(supplier: str) -> SupplierParserProfile:
    return SupplierParserProfile(
        supplier=supplier,
        row_selectors=("tr", ".product-item", ".item", ".variant", ".price-row", ".package", ".pack-size", ".sku-row", "[data-price]", "[data-price-amount]", "[data-final-price]", "[data-product-id]", "[data-sku]"),
        option_selectors=("option", "select option", "[data-price]", "[data-price-amount]", "[data-final-price]"),
        table_markers=("size", "pack", "price", "stock", "availability", "qty", "quantity", "catalog"),
        size_headers=("size", "pack", "package", "amount", "qty", "quantity", "unit"),
        price_headers=("price", "usd", "amount", "cost", "your price", "list price", "unit price"),
        stock_headers=("stock", "availability", "lead", "ships", "global stock", "usa stock"),
        product_path_hints=("product", "products", "compound", "item", "catalog", "shop"),
        parser_notes="Generic profile-backed parser.",
    )

def _profile(supplier: str, **kwargs: Any) -> SupplierParserProfile:
    data = _default_profile(supplier).__dict__.copy(); data.update(kwargs); return SupplierParserProfile(**data)

SUPPLIER_PARSER_PROFILES: dict[str, SupplierParserProfile] = {
    "TargetMol": _profile("TargetMol", row_selectors=("tr", ".sku-row", ".price-table tr", ".product-pack tr", ".product-info tr", "[data-price]"), table_markers=("pack size", "price", "usa stock", "global stock", "stock"), product_path_hints=("compound", "product"), parser_notes="Pack Size / Price / USA Stock / Global Stock table parser."),
    "MedChemExpress": _profile("MedChemExpress", row_selectors=("tr", ".product-price tr", ".table-price tr", ".package tr", ".size-price tr", "[data-price]"), table_markers=("size", "price", "stock", "purity"), product_path_hints=("cas", "compound", "product"), parser_notes="MCE size/price/stock product-table parser."),
    "SelleckChem": _profile("SelleckChem", row_selectors=("tr", ".price-table tr", ".product-price tr", ".size_price tr", ".packaging tr", "[data-price]"), table_markers=("pack size", "price", "stock", "qty"), product_path_hints=("products", "compound"), parser_notes="Selleck pack-size/price row parser."),
    "Cayman Chemical": _profile("Cayman Chemical", row_selectors=("tr", ".product-price tr", ".item-price tr", ".pricing tr", ".variant tr", "[data-price]"), table_markers=("item", "size", "price", "availability"), product_path_hints=("product", "item"), parser_notes="Cayman item/size/availability parser."),
    "MolPort": _profile("MolPort", row_selectors=("tr", ".packing", ".offer", ".seller-offer", ".price", "[data-price]"), table_markers=("available packings", "supplier", "amount", "price", "shipping"), product_path_hints=("shop/compound", "compound"), parser_notes="MolPort marketplace offer/packing parser."),
    "Adooq": _profile("Adooq", row_selectors=("tr", ".item", ".product-item", ".price-box", ".super-attribute-select option", "[data-price]"), table_markers=("grouped product items", "size", "price", "stock", "qty"), product_path_hints=("product", "catalog", ".html"), parser_notes="Adooq Magento-style grouped product items parser."),
    "ApexBio": _profile("ApexBio", row_selectors=("tr", ".price-table tr", ".product-price tr", ".package tr", ".size-price tr", "[data-price]"), table_markers=("size", "price", "stock", "qty"), product_path_hints=("products", "catalog"), parser_notes="ApexBio rows like 10mg / $67 / In stock."),
    "GLP Bio": _profile("GLP Bio", product_path_hints=("product", "catalog"), parser_notes="GLP Bio multilingual-safe product row parser."),
    "AbMole": _profile("AbMole", product_path_hints=("product", "catalog"), parser_notes="AbMole parser with shipping-threshold price noise rejection."),
    "ChemFaces": _profile("ChemFaces", product_path_hints=("products", "product"), parser_notes="ChemFaces natural-products price-row parser."),
    "BioCrick": _profile("BioCrick", product_path_hints=("products", "product"), parser_notes="BioCrick natural-products price-row parser."),
    "CSNpharm": _profile("CSNpharm", product_path_hints=("products", "product"), parser_notes="CSNpharm size/price/stock parser."),
    "InvivoChem": _profile("InvivoChem", product_path_hints=("product", "catalog"), parser_notes="InvivoChem Magento-like variant parser."),
    "AdooQ Bioscience": _profile("AdooQ Bioscience", product_path_hints=("product", "catalog"), parser_notes="Alternate AdooQ domain parser."),
    "Biorbyt": _profile("Biorbyt", product_path_hints=("product", "products"), parser_notes="Biorbyt reagent parser."),
    "Biosynth": _profile("Biosynth", row_selectors=("tr", ".product-row", ".pricing-row", ".price-row", ".variant", "[data-price]"), table_markers=("division", "code", "description", "spec", "pricing", "loading prices", "price"), product_path_hints=("/p/", "product"), parser_notes="Biosynth parser: product pages may show Loading Prices; product identity often lives in /p/CODE/CAS-slug URLs."),
    "US Biological": _profile("US Biological", product_path_hints=("product", "products", "catalog"), parser_notes="US Biological parser."),
    "1ClickChemistry": _profile("1ClickChemistry", product_path_hints=("product", "products", "catalog"), parser_notes="1ClickChemistry parser."),
    "TCI Chemicals": _profile("TCI Chemicals", row_selectors=("tr", ".product-list tr", ".price-table tr", ".sku-row", ".variant", "[data-price]"), table_markers=("packaging", "package", "price", "stock", "delivery"), product_path_hints=("product", "products"), parser_notes="TCI regional product-table parser."),
    "Oakwood Chemical": _profile("Oakwood Chemical", product_path_hints=("product", "products"), parser_notes="Oakwood specialty catalog parser."),
    "Chem-Impex": _profile("Chem-Impex", product_path_hints=("products", "product"), parser_notes="Chem-Impex catalog parser with marketing text filter."),
    "Combi-Blocks": _profile("Combi-Blocks", product_path_hints=("product", "products", "cgi-bin"), parser_notes="Combi-Blocks building-block parser."),
    "BLD Pharm": _profile("BLD Pharm", product_path_hints=("product", "products"), parser_notes="BLD Pharm building-block parser."),
    "Ambeed": _profile("Ambeed", row_selectors=("tr", ".variant", ".price", ".product-items tr", ".product-list tr", "[data-price]", "[data-sku]"), table_markers=("package", "size", "price", "stock", "ship", "login"), product_path_hints=("products", "product"), parser_notes="Ambeed parser: defaults to USA/USD headers; if location prompt remains, classify as location/login-gated rather than a parser miss."),
    "A2B Chem": _profile("A2B Chem", product_path_hints=("product", "products", "search.aspx"), parser_notes="A2B Chem parser."),
    "Enamine": _profile("Enamine", product_path_hints=("catalog", "product"), parser_notes="Enamine quote/account mixed parser."),
    "Matrix Scientific": _profile("Matrix Scientific", product_path_hints=("product", "products"), parser_notes="Matrix Scientific parser."),
    "Santa Cruz Biotechnology": _profile("Santa Cruz Biotechnology", product_path_hints=("product", "products"), parser_notes="SCBT product-row parser."),
    "CymitQuimica": _profile("CymitQuimica", product_path_hints=("products", "product"), parser_notes="Cymit marketplace parser."),
    "Toronto Research Chemicals": _profile("Toronto Research Chemicals", product_path_hints=("product", "products"), parser_notes="TRC reference/specialty parser."),
    "Fisher Scientific": _profile("Fisher Scientific", product_path_hints=("shop", "products", "catalog"), parser_notes="Fisher parser: public rows if present, otherwise account-price classification."),
    "Thermo Fisher / Alfa Aesar": _profile("Thermo Fisher / Alfa Aesar", product_path_hints=("products", "product", "search"), parser_notes="Thermo/Alfa session/account price classifier."),
    "Sigma-Aldrich": _profile("Sigma-Aldrich", product_path_hints=("product", "search"), parser_notes="Sigma country/account price classifier."),
    "VWR / Avantor": _profile("VWR / Avantor", product_path_hints=("store", "product", "search"), parser_notes="VWR account price classifier."),
    "ChemicalBook": _profile("ChemicalBook", product_path_hints=("chemicalproduct", "product"), parser_notes="Directory parser; RFQ leads, not source-of-truth pricing."),
    "ChemBlink": _profile("ChemBlink", product_path_hints=("products", "product"), parser_notes="Directory parser; RFQ lead parser."),
    "ChemExper": _profile("ChemExper", product_path_hints=("search", "product"), parser_notes="Directory parser; RFQ lead parser."),
    "LookChem": _profile("LookChem", product_path_hints=("cas", "product"), parser_notes="Directory parser; RFQ lead parser."),
}
for _adapter in ADAPTERS:
    SUPPLIER_PARSER_PROFILES.setdefault(_adapter.name, _default_profile(_adapter.name))

def _safe_float(value: Any) -> float | None:
    try:
        v = float(str(value).replace(',', '').replace('$', '').replace('USD', '').strip())
        return v if 0 < v < 10_000_000 else None
    except Exception:
        return None

def _normalize_unit(unit: str | None) -> str | None:
    if not unit: return None
    u = str(unit).strip().lower().replace('μ','u').replace('µ','u')
    return {'mcg':'ug','microgram':'ug','micrograms':'ug','ug':'ug','milligram':'mg','milligrams':'mg','mg':'mg','gram':'g','grams':'g','g':'g','kilogram':'kg','kilograms':'kg','kg':'kg','milliliter':'mL','milliliters':'mL','ml':'mL','liter':'L','liters':'L','l':'L'}.get(u, unit)

def _pack_is_reasonable(size: float | None, unit: str | None) -> bool:
    return bool(size is not None and unit is not None and 0 < size <= {'ug':1_000_000_000,'mg':1_000_000,'g':100_000,'kg':10_000,'mL':1_000_000,'L':10_000}.get(unit,100_000))

def _parse_pack(text: Any) -> tuple[float | None, str | None]:
    txt = str(text or '').replace('μ','u').replace('µ','u')
    sm = SOLUTION_RE.search(txt)
    if sm:
        size = _safe_float(sm.group(1) or sm.group(2)); return (size,'mL') if _pack_is_reasonable(size,'mL') else (None,None)
    m = PACK_RE.search(txt)
    if not m: return None, None
    size = _safe_float(m.group(1)); unit = _normalize_unit(m.group(2))
    return (size, unit) if _pack_is_reasonable(size, unit) else (None, None)

def _parse_price(text: Any, context: str = '') -> float | None:
    txt = str(text or '')
    if PRICE_NOISE_RE.search(f'{context} {txt}'[:1800]): return None
    m = PRICE_RE.search(txt)
    return _safe_float(m.group(1) or m.group(2)) if m else None

def _stock(text: Any) -> str:
    m = STOCK_RE.search(str(text or '')); return m.group(1).title() if m else 'Not visible'

def _form(pack_unit: str | None, text: Any) -> str:
    unit = str(pack_unit or '').strip()
    hay = str(text or '').lower()
    if 'reference standard' in hay or 'analytical standard' in hay or '(standard)' in hay:
        return 'standard/reference'
    # v19 regression guard: the actual purchased pack unit wins. A nearby DMSO
    # solution option in the same supplier table must not turn solid 100 mg /
    # 500 mg rows into solution rows.
    if unit in {'ug', 'mcg', 'mg', 'g', 'kg'}:
        return 'solid/mass'
    if unit in {'mL', 'ml', 'L', 'l'} or 'in dmso' in hay or re.search(r'\b\d+(?:\.\d+)?\s?mm\b', hay):
        return 'solution'
    return 'unknown'

def _row(profile: SupplierParserProfile, method: str, text: str, pack_size: float | None, pack_unit: str | None, price: float | None, confidence: str) -> dict[str, Any] | None:
    text = str(text or '')
    if price is None or not _pack_is_reasonable(pack_size, pack_unit) or PRICE_NOISE_RE.search(text):
        return None
    # Do not allow molecular weight/formula weight such as 534.55 g/mol to masquerade as a purchasable pack size.
    pack_token = f"{pack_size:g} {pack_unit}" if isinstance(pack_size, (int, float)) and pack_unit else ""
    idx = text.lower().find(pack_token.lower()) if pack_token else -1
    pack_window = text[max(0, idx - 5): idx + 45] if idx >= 0 else text[:120]
    if PACK_CONTEXT_NOISE_RE.search(pack_window):
        return None
    return {'method':f'supplier_parser:{profile.supplier}:{method}','pack_size':pack_size,'pack_unit':pack_unit,'price':price,'stock':_stock(text),'raw':[re.sub(r'\s+',' ',text).strip()[:1200]],'price_pairing_confidence':confidence,'product_form':_form(pack_unit,text),'supplier_parser_name':_parser_name(profile.supplier),'supplier_parser_status':'supplier_specific_price_rows_found'}

def _parse_html_tables(profile: SupplierParserProfile, soup: BeautifulSoup) -> list[dict[str, Any]]:
    out=[]
    for table in soup.find_all('table'):
        table_text=table.get_text(' ',strip=True).lower()
        if not any(m.lower() in table_text for m in profile.table_markers): continue
        headers=[]
        for tr in table.find_all('tr'):
            cells=[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])]
            if not cells: continue
            row_text=' | '.join(cells); lower=[c.lower() for c in cells]
            if any(any(h in c for h in profile.price_headers) for c in lower) and any(any(h in c for h in profile.size_headers) for c in lower):
                headers=lower; continue
            pack_text=row_text; price_text=row_text; stock_text=row_text
            if headers and len(headers)==len(cells):
                for h,c in zip(headers,cells):
                    if any(w in h for w in profile.size_headers): pack_text=c
                    if any(w in h for w in profile.price_headers): price_text=c
                    if any(w in h for w in profile.stock_headers): stock_text += ' '+c
            pack_size,pack_unit=_parse_pack(pack_text); price=_parse_price(price_text,row_text) or _parse_price(row_text,row_text)
            r=_row(profile,'table_row',f'{row_text} | {stock_text}',pack_size,pack_unit,price,'HIGH')
            if r: out.append(r)
    return out

def _parse_options(profile: SupplierParserProfile, soup: BeautifulSoup) -> list[dict[str, Any]]:
    out=[]
    for node in soup.select(','.join(profile.option_selectors)):
        attrs=' '.join(f'{k}={v}' for k,v in getattr(node,'attrs',{}).items()); text=f"{node.get_text(' ',strip=True)} {attrs}"
        pack_size,pack_unit=_parse_pack(text); price=_parse_price(text,text)
        if price is None:
            for key in ('data-price','data-price-amount','data-final-price','price','data-amount'):
                if node.has_attr(key): price=_safe_float(node.get(key)); break
        r=_row(profile,'option_or_data_attr',text,pack_size,pack_unit,price,'HIGH')
        if r: out.append(r)
    return out

def _walk_json(obj: Any):
    if isinstance(obj,dict):
        yield obj
        for v in obj.values(): yield from _walk_json(v)
    elif isinstance(obj,list):
        for item in obj: yield from _walk_json(item)

def _loads_json_candidates(text: str) -> list[Any]:
    txt=(text or '').strip(); candidates=[]
    if txt.startswith('{') or txt.startswith('['): candidates.append(txt)
    for m in re.finditer(r'(?s)(\{[^{}]{0,3000}(?:price|finalPrice|unitPrice|pack|size|sku)[^{}]{0,3000}\})', txt[:250000], re.I): candidates.append(m.group(1))
    parsed=[]
    for raw in candidates[:100]:
        try: parsed.append(json.loads(raw))
        except Exception: pass
    return parsed

def _parse_json(profile: SupplierParserProfile, soup: BeautifulSoup) -> list[dict[str, Any]]:
    out=[]
    for script in soup.find_all('script'):
        script_text=script.get_text(' ',strip=True)
        if not script_text or 'price' not in script_text.lower(): continue
        for root in _loads_json_candidates(script_text):
            for node in _walk_json(root):
                if not isinstance(node,dict): continue
                raw=json.dumps(node,ensure_ascii=False)[:1800]
                pack_sources=[v for k,v in node.items() if JSON_PACK_KEY_RE.search(str(k))]+[raw]
                price_sources=[v for k,v in node.items() if JSON_PRICE_KEY_RE.search(str(k))]+[raw]
                pack_size=pack_unit=price=None
                for src in pack_sources:
                    pack_size,pack_unit=_parse_pack(src)
                    if pack_size: break
                for src in price_sources:
                    price=_safe_float(src) if not isinstance(src,str) else _parse_price(src,raw)
                    if price: break
                r=_row(profile,'embedded_json',raw,pack_size,pack_unit,price,'HIGH')
                if r: out.append(r)
    return out

def _parse_row_containers(profile: SupplierParserProfile, soup: BeautifulSoup) -> list[dict[str, Any]]:
    out=[]
    for node in soup.select(','.join(profile.row_selectors)):
        text=node.get_text(' | ',strip=True)
        if len(text)<4: text=' '.join(f'{k}={v}' for k,v in getattr(node,'attrs',{}).items())
        if not text or PRICE_NOISE_RE.search(text): continue
        pack_size,pack_unit=_parse_pack(text); price=_parse_price(text,text)
        row_conf = 'HIGH' if profile.supplier in STRUCTURED_LADDER_SUPPLIERS and re.search(r'(?i)(in\s*stock|stock|available|qty|quantity|pack|size|price)', text) else 'MEDIUM'
        r=_row(profile,'row_container',text,pack_size,pack_unit,price,row_conf)
        if r: out.append(r)
    return out


def _parse_verified_price_ladder_block(profile: SupplierParserProfile, text: str) -> list[dict[str, Any]]:
    """Parse rendered catalog ladders as HIGH-confidence same-table rows.

    This covers price-first suppliers that flatten tables into text, for example
    TargetMol/Apex/Adooq-style "Pack Size Price Stock ... 500 mg $688 In Stock"
    and MedChemExpress-style "5 mg, USD 50, In-stock" rows.
    """
    clean = re.sub(r'\s+', ' ', (text or '').replace('μ', 'u').replace('µ', 'u'))
    if not clean:
        return []
    header_re = re.compile(
        r'(?i)(grouped product items\s+size\s+price\s+stock|pack\s*size\s+price|size\s+price\s+stock|size\s*/\s*price\s*/\s*stock|available\s+packings|price\s+usa\s+stock|product\s+items\s+size\s+price)'
    )
    starts = [m.start() for m in header_re.finditer(clean)]
    if not starts:
        # MCE can expose "5 mg, USD 50, In-stock" without a clean table header.
        if str(profile.supplier).lower() != 'medchemexpress':
            return []
        starts = [0]
    row_re = re.compile(
        r'(?P<pack>(?:\d+(?:\.\d+)?\s?(?:ug|mcg|mg|g|kg|mL|ml|L|l)|\d+(?:\.\d+)?\s?mL\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mM(?:\s*\(?in\s*DMSO\)?)?|\d+(?:\.\d+)?\s?mM\s*(?:x|\*)\s*\d+(?:\.\d+)?\s?mL(?:\s*in\s*DMSO)?))'
        r'(?P<middle>.{0,100}?)'
        r'(?P<price>(?:USD|US\$|\$)\s*[0-9][0-9,]*(?:\.\d{1,4})?|[0-9][0-9,]*(?:\.\d{1,4})?\s*USD)'
        r'(?P<tail>.{0,120}?(?:in\s*-?\s*stock|out\s*of\s*stock|available|ships?\s*in\s*[^.;,|]{1,30}|estimated\s*time\s*of\s*arrival[^.;,|]{0,50})?)',
        re.I,
    )
    out: list[dict[str, Any]] = []
    for start in starts[:6]:
        window = clean[start:start + 12000]
        for m in row_re.finditer(window):
            raw = re.sub(r'\s+', ' ', f"{m.group('pack')} {m.group('middle') or ''} {m.group('price')} {m.group('tail') or ''}").strip()
            if not re.search(r'(?i)(\$|USD|US\$)', m.group('price') or ''):
                continue
            pack_size, pack_unit = _parse_pack(m.group('pack'))
            price = _parse_price(m.group('price'), raw)
            r = _row(profile, 'verified_ladder_block', raw, pack_size, pack_unit, price, 'HIGH')
            if r:
                out.append(r)
            if len(out) >= 60:
                return out
    return out

def _parse_text_ladders(profile: SupplierParserProfile, text: str) -> list[dict[str, Any]]:
    """Parse flattened supplier catalog ladders like:

    Pack Size Price USA Stock Global Stock
    5 mg$39 In Stock In Stock
    10 mg$64 In Stock In Stock

    or ApexBio/Magento-style:
    Grouped product items Size Price Stock Qty 10mg $67.00 In stock ...

    These rows are supplier-structured price ladders even if the rendered HTML is
    flattened to text by requests/BeautifulSoup, so we mark them HIGH for known
    structured-ladder suppliers.
    """
    clean = re.sub(r'\s+', ' ', (text or '').replace('μ', 'u').replace('µ', 'u'))
    if not clean:
        return []
    marker_re = re.compile(
        r'(?i)(pack\s*size\s*price|size\s*/?\s*price|size\s+price|available\s+packings|price\s+stock|product\s+items|grouped\s+product\s+items|usa\s+stock\s+global\s+stock)'
    )
    markers = [m.start() for m in marker_re.finditer(clean)]
    if not markers:
        return []
    out: list[dict[str, Any]] = []
    # Stop windows before calculator/formulation zones to avoid solubility or dosing rows.
    stop_re = re.compile(r'(?i)(solution preparation table|molarity calculator|dilution calculator|in vivo formulation|featured recommendations|related products|customers also)')
    row_re = re.compile(
        r'(?P<pack>(?:\d+(?:\.\d+)?\s?(?:ug|mcg|mg|g|kg|mL|ml|L|l)|\d+(?:\.\d+)?\s?mL\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mM(?:\s*\(?in\s*DMSO\)?)?|\d+(?:\.\d+)?\s?mM\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mL(?:\s*in\s*DMSO)?))'
        r'(?P<middle>.{0,180}?)'
        r'(?P<price>(?:USD|US\$|\$)\s*[0-9][0-9,]*(?:\.\d{1,4})?|[0-9][0-9,]*(?:\.\d{1,4})?\s*USD)'
        r'(?P<tail>.{0,120}?)'
        r'(?=(?:\d+(?:\.\d+)?\s?(?:ug|mcg|mg|g|kg|mL|ml|L|l)|$))',
        re.I,
    )
    for start in markers[:8]:
        window = clean[start:start + 12000]
        sm = stop_re.search(window)
        if sm:
            window = window[:sm.start()]
        header = clean[start:start + 250]
        header_conf = 'HIGH' if (profile.supplier in STRUCTURED_LADDER_SUPPLIERS or re.search(r'(?i)(pack\s*size|grouped\s+product\s+items|usa\s+stock)', header)) else 'MEDIUM'
        for m in row_re.finditer(window):
            raw = window[max(0, m.start() - 90):min(len(window), m.end() + 180)]
            # Guard against formulation/calculator rows. They contain mg/mL or mM quantities
            # but not purchasable pack/price rows.
            if re.search(r'(?i)(molarity|calculator|dilution|dose|formula|molecular\s+weight|g/mol|solution\s+preparation)', raw):
                continue
            pack_size, pack_unit = _parse_pack(m.group('pack'))
            price = _parse_price(m.group('price'), raw)
            r = _row(profile, 'structured_text_ladder', raw, pack_size, pack_unit, price, header_conf)
            if r:
                out.append(r)
    return out

def _parse_adjacent_text_pairs(profile: SupplierParserProfile, text: str) -> list[dict[str, Any]]:
    """Find pack/price pairs in supplier page text when table labels are absent.

    This is intentionally LOW confidence because it is not a formal table row, but it is
    useful for suppliers such as Ambeed or marketplace pages where public prices are
    rendered as adjacent text after a location/country widget.
    """
    clean = re.sub(r'\s+', ' ', (text or '').replace('μ', 'u').replace('µ', 'u'))
    if not clean:
        return []
    out: list[dict[str, Any]] = []
    pack_price = re.compile(
        r'(?P<pack>\d+(?:\.\d+)?\s?(?:ug|mcg|mg|g|kg|mL|ml|L|l))'
        r'(?P<middle>.{0,140}?)'
        r'(?P<price>(?:USD|US\$|\$)\s*[0-9][0-9,]*(?:\.\d{1,4})?|[0-9][0-9,]*(?:\.\d{1,4})?\s*USD)',
        re.I,
    )
    # Deliberately avoid price→pack pairing. In catalog tables, a price is commonly
    # followed by stock text and then the next row's pack size, which creates false
    # pairs such as "$39 In Stock 10 mg".
    for rx, direction in [(pack_price, 'adjacent_pack_price')]:
        for m in rx.finditer(clean[:180000]):
            middle = m.group('middle') or ''
            if len(middle) > 140 or PRICE_NOISE_RE.search(middle):
                continue
            raw = clean[max(0, m.start()-100):min(len(clean), m.end()+180)]
            # Guard CAS-number fragments such as "487-41-2 Grouped..." being read
            # as a fake "2 g" pack. Adjacent parsing is a fallback; structured
            # ladder parsing above should handle real rows.
            pre = clean[max(0, m.start()-35):m.start()]
            if re.search(r'\d{2,7}-\d{2}-$', pre) or re.search(r'(?i)cas\s*(?:no\.?|number|#)?[^.;|]{0,80}$', pre):
                continue
            if re.search(r'(?i)^\s*\d+(?:\.\d+)?\s?g\s*(?:rouped|lobal|eneric)', m.group('pack') + (m.group('middle') or '')[:10]):
                continue
            pack_size, pack_unit = _parse_pack(m.group('pack'))
            price = _parse_price(m.group('price'), raw)
            r = _row(profile, direction, raw, pack_size, pack_unit, price, 'LOW')
            if r:
                out.append(r)
            if len(out) >= 40:
                return out
    return out


def _parse_js_regex_pairs(profile: SupplierParserProfile, soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    key_pair_res = [
        re.compile(r"""(?is)(?:sku|size|pack|package|qty|quantity|label|name)["']?\s*[:=]\s*["']?(?P<pack>\d+(?:\.\d+)?\s?(?:ug|mcg|mg|g|kg|mL|ml|L|l))[^{}\[\]]{0,240}?(?:price|finalPrice|unitPrice|salePrice|regularPrice|yourPrice)["']?\s*[:=]\s*["']?\$?(?P<price>[0-9][0-9,]*(?:\.\d{1,4})?)"""),
        re.compile(r"""(?is)(?:price|finalPrice|unitPrice|salePrice|regularPrice|yourPrice)["']?\s*[:=]\s*["']?\$?(?P<price>[0-9][0-9,]*(?:\.\d{1,4})?)[^{}\[\]]{0,240}?(?:sku|size|pack|package|qty|quantity|label|name)["']?\s*[:=]\s*["']?(?P<pack>\d+(?:\.\d+)?\s?(?:ug|mcg|mg|g|kg|mL|ml|L|l))"""),
    ]
    for script in soup.find_all('script'):
        script_text = script.get_text(' ', strip=True)
        if not script_text or not any(tok in script_text.lower() for tok in ['price', 'sku', 'pack', 'size', 'qty']):
            continue
        search_text = script_text[:350000]
        for rx in key_pair_res:
            for m in rx.finditer(search_text):
                raw = search_text[max(0, m.start()-120):min(len(search_text), m.end()+180)]
                pack_size, pack_unit = _parse_pack(m.group('pack'))
                price = _safe_float(m.group('price'))
                r = _row(profile, 'js_keyed_variant', raw, pack_size, pack_unit, price, 'HIGH')
                if r:
                    out.append(r)
                if len(out) >= 50:
                    return out[:50]
        # Use windows around price keys to catch visible JavaScript strings with $/USD.
        for m in re.finditer(r'(?i)(price|finalPrice|unitPrice|salePrice|regularPrice|yourPrice)', search_text):
            window = search_text[max(0, m.start()-600):min(len(search_text), m.end()+900)]
            out.extend(_parse_adjacent_text_pairs(profile, window))
            if len(out) >= 50:
                return out[:50]
    return out


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank={'HIGH':3,'MEDIUM':2,'LOW':1}; rows=sorted(rows,key=lambda r:rank.get(str(r.get('price_pairing_confidence')),0),reverse=True)
    seen=set(); out=[]
    for row in rows:
        key=(round(float(row.get('pack_size') or 0),10),str(row.get('pack_unit') or '').lower(),round(float(row.get('price') or 0),4),str(row.get('product_form') or ''))
        if key in seen: continue
        seen.add(key); out.append(row)
    return out

def _parser_name(supplier: str) -> str:
    return 'parse_'+re.sub(r'[^a-z0-9]+','_',supplier.lower()).strip('_')

def parser_name_for_supplier(supplier: str | None) -> str:
    supplier=supplier or 'Unknown supplier'; profile=SUPPLIER_PARSER_PROFILES.get(supplier,_default_profile(supplier)); return _parser_name(profile.supplier)

def extract_supplier_specific_rows(supplier: str | None, soup: BeautifulSoup, text: str, url: str='') -> tuple[list[dict[str, Any]], str, str]:
    supplier=supplier or 'Unknown supplier'; profile=SUPPLIER_PARSER_PROFILES.get(supplier,_default_profile(supplier)); name=_parser_name(profile.supplier)
    try:
        rows=[]; rows.extend(_parse_html_tables(profile,soup)); rows.extend(_parse_verified_price_ladder_block(profile,text)); rows.extend(_parse_text_ladders(profile,text)); rows.extend(_parse_options(profile,soup)); rows.extend(_parse_json(profile,soup)); rows.extend(_parse_js_regex_pairs(profile,soup)); rows.extend(_parse_row_containers(profile,soup)); rows.extend(_parse_adjacent_text_pairs(profile,text)); rows=_dedupe(rows)
    except Exception as exc:
        return [], name, f'supplier_specific_parser_failed:{type(exc).__name__}'
    if rows: return rows, name, 'supplier_specific_price_rows_found'
    low=(text or '')[:25000].lower()
    if LOCATION_PROMPT_RE.search(low): return [], name, 'supplier_specific_location_prompt_us_default_attempted_possible_login_price'
    if any(t in low for t in ['loading prices','prices are taking a little longer','retry pricing','price loader']): return [], name, 'supplier_specific_js_price_loader_detected'
    if any(t in low for t in ['sign in','login','your price','account price','register to view','see vip prices','create an account']): return [], name, 'supplier_specific_checked_login_or_account_price'
    if any(t in low for t in ['request quote','request a quote','price on request','please inquire','inquire']): return [], name, 'supplier_specific_checked_quote_or_inquiry'
    return [], name, 'supplier_specific_checked_no_public_price_rows'

def supplier_specific_variant_rows(supplier: str | None, soup: BeautifulSoup, text: str) -> list[dict[str, Any]]:
    rows, _, _ = extract_supplier_specific_rows(supplier, soup, text, '')
    return rows

def supplier_parser_status(supplier: str | None, rows_found: int, page_text: str='') -> str:
    if rows_found>0: return 'supplier_specific_price_rows_found'
    low=(page_text or '')[:25000].lower()
    if LOCATION_PROMPT_RE.search(low): return 'supplier_specific_location_prompt_us_default_attempted_possible_login_price'
    if any(t in low for t in ['loading prices','prices are taking a little longer','retry pricing','price loader']): return 'supplier_specific_js_price_loader_detected'
    if any(t in low for t in ['sign in','login','your price','account price','register to view','see vip prices','create an account']): return 'supplier_specific_checked_login_or_account_price'
    if any(t in low for t in ['request quote','request a quote','price on request','please inquire','inquire']): return 'supplier_specific_checked_quote_or_inquiry'
    return 'supplier_specific_checked_no_public_price_rows'

def supplier_parser_registry_report() -> list[dict[str, str]]:
    return [{'supplier':a.name,'source_tier':a.source_tier,'expected_behavior':a.expected_behavior,'parser':parser_name_for_supplier(a.name),'parser_profile':SUPPLIER_PARSER_PROFILES[a.name].parser_notes} for a in ADAPTERS]
