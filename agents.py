"""Three domain-expert agents + orchestrator.
Each agent = deterministic analytics 'tools' (Pandas/NumPy) + an LLM reasoning layer (optional).
Maps 1:1 to LangGraph nodes or CrewAI agents (see README)."""
import os, numpy as np, pandas as pd
from math import sqrt
from statistics import NormalDist

# ---------- optional LLM layer ----------
def llm(system, prompt, fallback):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return fallback
    try:
        import anthropic
        r = anthropic.Anthropic(api_key=key).messages.create(
            model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=700,
            system=system, messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    except Exception as e:
        return fallback + f"\n(LLM unavailable: {e})"

# =====================================================================
class DemandForecastAgent:
    name, role = "Demand Forecasting & Sales Trend Agent", "Lead Predictive Analyst"
    SYS = "You are a retail demand-planning expert. Be concise, quantitative, flag risks."

    def seasonal_index(self, y, period=52):
        n = len(y) // period * period
        if n < period:
            return np.ones(period)
        m = y[-n:].reshape(-1, period)
        idx = (m / m.mean(axis=1, keepdims=True)).mean(axis=0)
        return idx / idx.mean()

    def forecast_series(self, y, h, ext=1.0, period=52):
        idx = self.seasonal_index(y, period)
        pos = np.arange(len(y)) % period
        des = y / np.maximum(idx[pos], 0.05)
        k = 26
        slope, icpt = np.polyfit(np.arange(k), des[-k:], 1)
        fut = np.arange(len(y), len(y) + h)
        lvl = icpt + slope * (k - 1 + np.arange(1, h + 1))
        return np.maximum(0, lvl * idx[fut % period] * ext)

    def run(self, state):
        cfg, sales = state["config"], state["data"]["sales"]
        h, out = cfg["horizon"], []
        for sku, g in sales.groupby("sku"):
            y = g.sort_values("week").units.values.astype(float)
            # back-test on last h weeks
            bt = self.forecast_series(y[:-h], h)
            actual = y[-h:]
            mape = float(np.mean(np.abs(actual - bt) / np.maximum(actual, 1)))
            sigma = float(np.std(actual - bt, ddof=1))
            fc = self.forecast_series(y, h, cfg["external_index"])
            out.append(dict(sku=sku, avg_weekly_fc=fc.mean(), fc_total=fc.sum(),
                            recent_avg=y[-8:].mean(), prior_avg=y[-16:-8].mean(),
                            sigma_err=sigma, mape=mape))
        df = pd.DataFrame(out)
        df["trend_pct"] = (df.recent_avg / df.prior_avg - 1) * 100
        df["trend"] = np.where(df.trend_pct > 5, "Rising", np.where(df.trend_pct < -5, "Falling", "Stable"))
        state["forecast"] = df
        fb = (f"Forecast for next {h} weeks across {len(df)} SKUs. Mean back-test MAPE "
              f"{df.mape.mean():.1%}. Rising: {(df.trend=='Rising').sum()}, Falling: {(df.trend=='Falling').sum()}. "
              f"External demand index applied: {cfg['external_index']:.2f}.")
        state["narrative"]["forecast"] = llm(self.SYS, fb + "\n" + df.round(1).to_string(), fb)
        return state

# =====================================================================
class WorkingCapitalAgent:
    name, role = "Working Capital & Inventory Allocation Agent", "Inventory & Financial Controller"
    SYS = "You are a retail inventory/finance controller. Explain trade-offs between service level and cash."

    def run(self, state):
        cfg, d = state["config"], state["data"]
        z = NormalDist().inv_cdf(cfg["service_level"])
        df = (state["forecast"].merge(d["sku"], on="sku").merge(d["inventory"], on="sku")
              .merge(d["suppliers"], on="supplier"))
        wk = df.avg_weekly_fc
        L, sL = df.lead_mean_wks, df.lead_std_wks
        # safety stock covers demand error over lead time AND lead-time variability
        df["safety_stock"] = z * np.sqrt(L * df.sigma_err**2 + (wk**2) * sL**2)
        df["rop"] = wk * L + df.safety_stock
        annual_d = wk * 52
        h_cost = cfg["holding_rate"] * df.unit_cost
        df["eoq"] = np.sqrt(2 * annual_d * cfg["order_cost"] / h_cost)
        df["position"] = df.on_hand + df.on_order
        df["weeks_cover"] = df.on_hand / np.maximum(wk, 1)
        df["turnover"] = annual_d / np.maximum(df.on_hand, 1)
        df["stock_value"] = df.on_hand * df.unit_cost
        df["deficit"] = np.maximum(0, df.rop - df.position)
        # ABC by annual revenue-at-cost
        val = annual_d * df.unit_cost
        cum = val.sort_values(ascending=False).cumsum() / val.sum()
        df["abc"] = cum.reindex(df.index).map(lambda x: "A" if x <= .7 else ("B" if x <= .9 else "C"))
        # Cash conversion cycle
        dio = 365 / max(float((annual_d * df.unit_cost).sum() / max(df.stock_value.sum(), 1)), 1e-9)
        dpo = float(np.average(df.payment_days, weights=df.stock_value + 1)) if cfg.get("use_supplier_dpo") else cfg["dpo"]
        state["ccc"] = dict(DIO=dio, DSO=cfg["dso"], DPO=dpo, CCC=dio + cfg["dso"] - dpo,
                            inventory_value=float(df.stock_value.sum()),
                            carrying_cost_annual=float(df.stock_value.sum() * cfg["holding_rate"]))
        # exceptions
        exc = []
        for _, r in df.iterrows():
            if r.weeks_cover > cfg["max_weeks_cover"] or r.turnover < cfg["min_turnover"]:
                exc.append(dict(sku=r.sku, type="Slow-moving / overstock", severity="High" if r.abc != "C" else "Medium",
                                detail=f"{r.weeks_cover:.1f} wks cover, {r.turnover:.1f} turns/yr, Rs {r.stock_value:,.0f} tied up",
                                action="Pause reorders; markdown or transfer stock"))
            if r.on_hand < r.safety_stock:
                exc.append(dict(sku=r.sku, type="Stockout risk", severity="High",
                                detail=f"On hand {r.on_hand:.0f} < safety stock {r.safety_stock:.0f}",
                                action="Expedite replenishment"))
        state["inv"] = df
        state["exceptions"] = exc
        c = state["ccc"]
        fb = (f"Inventory Rs {c['inventory_value']:,.0f}; DIO {c['DIO']:.0f}d, DSO {c['DSO']}d, DPO {c['DPO']:.0f}d "
              f"=> cash conversion cycle {c['CCC']:.0f} days. Annual carrying cost Rs {c['carrying_cost_annual']:,.0f}. "
              f"{(df.deficit>0).sum()} SKUs are at/below reorder point.")
        state["narrative"]["working_capital"] = llm(self.SYS, fb, fb)
        return state

# =====================================================================
class ProcurementAgent:
    name, role = "Supplier Negotiation & Procurement Agent", "Procurement Manager"
    SYS = "You are a retail procurement manager. Write short, firm, professional supplier proposals."

    def run(self, state):
        cfg, df, sup = state["config"], state["inv"], state["data"]["suppliers"]
        pos, skipped = [], []
        cand = df[df.deficit > 0].copy()
        # slow-movers are never reordered; priority = class A first, then biggest relative shortfall
        cand["prio"] = cand.abc.map({"A": 0, "B": 1, "C": 2})
        cand["short_ratio"] = cand.deficit / np.maximum(cand.rop, 1)
        cand = cand.sort_values(["prio", "short_ratio"], ascending=[True, False])
        budget = cfg["wc_budget"]
        for _, r in cand.iterrows():
            target = r.rop + r.eoq
            qty = max(r.eoq, target - r.position)
            qty = float(np.ceil(qty))
            disc = 0.0
            if qty >= r.bulk_break_units:
                disc = r.bulk_discount
            elif qty >= 0.7 * r.bulk_break_units:           # nudge up to capture the price break if cheap to do so
                extra_cost = (r.bulk_break_units - qty) * r.unit_cost * r.price_index
                saving = r.bulk_break_units * r.unit_cost * r.price_index * r.bulk_discount
                if saving > extra_cost * cfg["holding_rate"] * (r.lead_mean_wks + 4) / 52 and r.weeks_cover < 4:
                    qty, disc = float(r.bulk_break_units), r.bulk_discount
            price = r.unit_cost * r.price_index * (1 - disc)
            cost = qty * price
            if cost > budget:
                qty = float(np.floor(budget / price))
                cost = qty * price
                if qty < max(1, 0.25 * r.deficit):
                    skipped.append(dict(sku=r.sku, reason="Working-capital budget exhausted", needed=r.deficit))
                    continue
            budget -= cost
            risk = r.reliability < cfg["min_reliability"] or r.lead_std_wks > 1.0
            pos.append(dict(po=f"PO-{len(pos)+1:03d}", sku=r.sku, abc=r.abc, supplier=r.supplier, qty=qty,
                            unit_price=round(price, 2), po_value=round(cost, 2), bulk_discount=disc,
                            exp_lead_wks=r.lead_mean_wks, reliability=r.reliability,
                            delay_risk="High" if risk else "Low",
                            order_by_buffer_wks=round(r.lead_std_wks * 1.65, 1) if risk else 0.0))
        po_df = pd.DataFrame(pos)
        state["pos"], state["skipped"] = po_df, pd.DataFrame(skipped)
        # delay-risk exceptions + backup-supplier suggestion
        for _, p in (po_df[po_df.delay_risk == "High"].iterrows() if len(po_df) else []):
            alt = sup[(sup.supplier != p.supplier) & (sup.reliability >= cfg["min_reliability"])].sort_values("lead_mean_wks")
            state["exceptions"].append(dict(
                sku=p.sku, type="Supplier delay risk", severity="High",
                detail=f"{p.supplier} reliability {p.reliability:.0%}, lead {p.exp_lead_wks} wks (variable)",
                action=(f"Place order {p.order_by_buffer_wks} wks earlier; split volume with {alt.iloc[0].supplier}"
                        if len(alt) else "Place order earlier")))
        # negotiation drafts: top 3 suppliers by spend
        drafts = {}
        if len(po_df):
            for s, g in po_df.groupby("supplier").po_value.sum().sort_values(ascending=False).head(3).items():
                info = sup[sup.supplier == s].iloc[0]
                fb = (f"Subject: Volume commitment - {s}\n\nWe plan to place Rs {g:,.0f} of orders this cycle. "
                      f"We would like to (1) extend the {info.bulk_discount:.0%} volume discount to orders from "
                      f"{int(info.bulk_break_units*0.7)} units, (2) move payment terms from {int(info.payment_days)} to "
                      f"{int(info.payment_days)+15} days, and (3) agree a delivery-delay penalty clause "
                      f"(current on-time reliability: {info.reliability:.0%}). In return we offer a 6-month rolling forecast.")
                drafts[s] = llm(self.SYS, "Draft a negotiation email:\n" + fb, fb)
        state["negotiation"] = drafts
        tot = po_df.po_value.sum() if len(po_df) else 0
        fb = f"{len(po_df)} POs worth Rs {tot:,.0f} generated; {len(skipped)} deferred for budget. Budget left Rs {budget:,.0f}."
        state["narrative"]["procurement"] = llm(self.SYS, fb, fb)
        return state

# =====================================================================
class Orchestrator:
    """Linear LangGraph-style graph: forecast -> working_capital -> procurement -> synthesis."""
    def __init__(self):
        self.nodes = [DemandForecastAgent(), WorkingCapitalAgent(), ProcurementAgent()]

    def run(self, data, config):
        state = dict(data=data, config=config, narrative={}, exceptions=[], log=[])
        for n in self.nodes:
            try:
                state = n.run(state)
                state["log"].append(f"OK  {n.name}")
            except Exception as e:                      # Step 3: automated exception handling
                state["log"].append(f"ERR {n.name}: {e}")
                raise
        state["plan"] = self.synthesize(state)
        return state

    def synthesize(self, s):
        c, po, inv = s["ccc"], s["pos"], s["inv"]
        slow = [e for e in s["exceptions"] if e["type"].startswith("Slow")]
        locked = sum(inv.loc[inv.sku.isin([e["sku"] for e in slow]), "stock_value"])
        plan = [
            f"1. APPROVE {len(po)} purchase orders totalling Rs {po.po_value.sum() if len(po) else 0:,.0f} "
            f"(budget Rs {s['config']['wc_budget']:,.0f}).",
            f"2. RELEASE cash: {len(slow)} slow-moving SKUs hold Rs {locked:,.0f}; freeze reorders and run markdown/transfer.",
            f"3. MITIGATE supplier risk on {sum(e['type']=='Supplier delay risk' for e in s['exceptions'])} POs "
            f"(order earlier / dual-source).",
            f"4. TARGET cash conversion cycle: now {c['CCC']:.0f} days (DIO {c['DIO']:.0f}).",
            f"5. NEGOTIATE with {len(s['negotiation'])} suppliers using drafted proposals.",
        ]
        fb = "\n".join(plan)
        return llm("You are the CFO-level synthesizer. Merge into a weekly action plan.",
                   fb + "\n\n" + "\n".join(s["narrative"].values()), fb)
