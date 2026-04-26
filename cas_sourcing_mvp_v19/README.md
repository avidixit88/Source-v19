# CAS Sourcing & Procurement Intelligence MVP v19

Streamlit MVP for CAS-number sourcing/procurement workflows.

## v19 focus

v19 builds on v18 with a MedChemExpress-focused adapter fix while preserving the stable suppliers:

- Card-level CAS-gated product link expansion so search-page CAS noise does not validate unrelated product links.
- Wrong-CAS product identity rejection when a product page declares a different CAS in its identity block.
- Supplier search pages are treated as navigation/discovery pages, not verified price pages.
- Supplier-specific parser profiles are still used for quantity/price extraction once a product page is reached.
- Deferred product-name retry pass: after any supplier confirms the product name, earlier suppliers with weak CAS search pages can be retried using product-name probes.
- Product-level evidence exports include identity reason, observed CAS numbers, and price lead type for auditability.
- MedChemExpress product pages receive a stronger US/browser fetch context.
- If MCE blocks a direct fetch, v19 can recover public MCE pack/price rows from reader text or optional SerpAPI snippets and labels that path explicitly.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push the full folder structure to GitHub and deploy `app.py` on Streamlit Cloud.
Use Python 3.12 for the most stable deployment path.


## v19 regression guards

- Mass pack rows now win over nearby DMSO solution text, so largest solid packs stay model-eligible.
- Supplier/MCE price regex handles both `$50` and `USD 50` formats.
- Verified ladder rows from price-first suppliers are parsed as HIGH-confidence same-table pairs.
- Charts keep candidate price leads visible while the desired-quantity model only uses verified CAS/product rows.
