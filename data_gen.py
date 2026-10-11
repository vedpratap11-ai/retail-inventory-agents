"""Step 1 - Data ingestion: synthetic POS, inventory and supplier data (swap for real CSVs)."""
import numpy as np, pandas as pd

def make_data(seed=7, weeks=104):
    rng = np.random.default_rng(seed)
    cats = {"Staples": 0.15, "Beverages": 0.3, "Personal Care": 0.1, "Snacks": 0.25, "Seasonal": 0.6}
    skus = []
    for i in range(1, 21):
        cat = list(cats)[i % 5]
        skus.append(dict(sku=f"SKU{i:03d}", category=cat, unit_cost=float(rng.integers(40, 900)),
                         base_demand=float(rng.integers(40, 400)), season_amp=cats[cat],
                         trend=float(rng.normal(0.002, 0.003)),
                         supplier=f"SUP{rng.integers(1, 5)}"))
    sku_df = pd.DataFrame(skus)
    t = np.arange(weeks)
    rows = []
    for s in skus:
        season = 1 + s["season_amp"] * np.sin(2 * np.pi * (t - rng.integers(0, 52)) / 52)
        mu = s["base_demand"] * season * (1 + s["trend"] * t)
        units = np.maximum(0, rng.normal(mu, 0.12 * mu)).round()
        for w in range(weeks):
            rows.append((w, s["sku"], units[w]))
    sales = pd.DataFrame(rows, columns=["week", "sku", "units"])
    # inventory: some SKUs deliberately overstocked / understocked
    inv = sku_df[["sku"]].copy()
    avg = sales.groupby("sku").units.mean().reindex(inv.sku).values
    inv["on_hand"] = (avg * rng.choice([0.4, 0.9, 1.5, 6, 16], len(inv))).round()
    inv["on_order"] = (avg * rng.choice([0, 0, 1], len(inv))).round()
    sup = pd.DataFrame({
        "supplier": [f"SUP{i}" for i in range(1, 5)],
        "lead_mean_wks": [1.0, 2.0, 3.0, 1.5],
        "lead_std_wks": [0.2, 0.5, 1.2, 0.4],
        "reliability": [0.97, 0.92, 0.78, 0.90],
        "price_index": [1.00, 0.96, 0.92, 0.98],
        "bulk_break_units": [500, 800, 1200, 600],
        "bulk_discount": [0.03, 0.04, 0.06, 0.03],
        "payment_days": [30, 45, 60, 30],
    })
    return dict(sku=sku_df, sales=sales, inventory=inv, suppliers=sup)

DEFAULT_CONFIG = dict(
    holding_rate=0.22,        # annual carrying cost as % of inventory value (capital + storage + shrink)
    order_cost=1500.0,        # fixed cost per purchase order (INR)
    service_level=0.95,
    wc_budget=6_000_000.0,    # max cash for this week's POs (INR)
    max_weeks_cover=12,       # slow-mover threshold
    min_turnover=4.0,         # annual turns below this -> exception
    min_reliability=0.85,
    dso=12, dpo=38,           # receivable / payable days
    external_index=1.00,      # e.g. 1.10 = festive / promo uplift
    horizon=8,
)
