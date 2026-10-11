import streamlit as st, pandas as pd
from data_gen import make_data, DEFAULT_CONFIG
from agents import Orchestrator

st.set_page_config(page_title="Retail Inventory Agents", layout="wide")
st.title("Retail inventory and working capital planner")
st.caption("Three agents forecast demand, size stock against cash, and draft purchase orders. You approve.")

with st.sidebar:
    st.header("Constraints")
    cfg = dict(DEFAULT_CONFIG)
    cfg["wc_budget"] = st.number_input("Weekly PO budget (Rs)", 100_000, 20_000_000, int(cfg["wc_budget"]), 100_000)
    cfg["service_level"] = st.slider("Service level", 0.80, 0.995, cfg["service_level"], 0.005)
    cfg["holding_rate"] = st.slider("Annual carrying rate", 0.08, 0.40, cfg["holding_rate"], 0.01)
    cfg["external_index"] = st.slider("External demand index (festive / promo)", 0.7, 1.5, 1.0, 0.05)
    cfg["max_weeks_cover"] = st.slider("Slow-mover threshold (weeks of cover)", 4, 26, cfg["max_weeks_cover"])
    up = st.file_uploader("Replace data (sales.csv: week, sku, units)", type="csv")

@st.cache_data
def load(): return make_data()
data = load()
if up: data["sales"] = pd.read_csv(up)

if "approved" not in st.session_state: st.session_state.approved = set()
S = Orchestrator().run(data, cfg)

c = S["ccc"]
m = st.columns(5)
m[0].metric("Inventory value", f"Rs {c['inventory_value']/1e5:,.1f} L")
m[1].metric("Cash conversion cycle", f"{c['CCC']:.0f} days")
m[2].metric("Annual carrying cost", f"Rs {c['carrying_cost_annual']/1e5:,.1f} L")
m[3].metric("Open POs", len(S["pos"]))
m[4].metric("Exceptions", len(S["exceptions"]))

t = st.tabs(["Weekly action plan", "Demand forecast", "Working capital", "Purchase orders", "Exceptions", "Supplier proposals"])
with t[0]:
    st.text(S["plan"])
    with st.expander("Agent notes"):
        for k, v in S["narrative"].items(): st.markdown(f"**{k.replace('_',' ').title()}**"); st.write(v)
        st.code("\n".join(S["log"]))
with t[1]:
    sku = st.selectbox("SKU", S["forecast"].sku)
    hist = data["sales"].query("sku==@sku").sort_values("week").set_index("week").units
    st.line_chart(hist.tail(52))
    st.dataframe(S["forecast"].round(2), use_container_width=True)
with t[2]:
    cols = ["sku","abc","on_hand","on_order","avg_weekly_fc","safety_stock","rop","eoq","weeks_cover","turnover","stock_value"]
    st.dataframe(S["inv"][cols].round(1), use_container_width=True)
    st.bar_chart(S["inv"].set_index("sku").weeks_cover)
with t[3]:
    po = S["pos"]
    if po.empty: st.info("No SKU is at or below its reorder point.")
    else:
        st.write(f"Total value: Rs {po.po_value.sum():,.0f} of Rs {cfg['wc_budget']:,.0f} budget")
        for _, r in po.iterrows():
            a, b = st.columns([5, 1])
            a.write(f"**{r.po}** {r.sku} (class {r.abc}) from {r.supplier}: {r.qty:.0f} units at Rs {r.unit_price:,.2f} = Rs {r.po_value:,.0f}"
                    f"  | delay risk: {r.delay_risk}" + (f" | bulk discount {r.bulk_discount:.0%}" if r.bulk_discount else ""))
            if b.button("Approve", key=r.po, disabled=r.po in st.session_state.approved):
                st.session_state.approved.add(r.po); st.rerun()
        st.success(f"Approved: {len(st.session_state.approved)} / {len(po)}")
    if len(S["skipped"]): st.warning("Deferred (budget)"); st.dataframe(S["skipped"])
with t[4]:
    ex = pd.DataFrame(S["exceptions"])
    st.dataframe(ex, use_container_width=True) if len(ex) else st.success("No exceptions.")
with t[5]:
    for s, txt in S["negotiation"].items():
        with st.expander(s): st.text_area("Draft", txt, height=220, key=s)
