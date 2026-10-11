# Retail Inventory & Working Capital Agents (Group 10)
Run: `pip install -r requirements.txt && streamlit run app.py`
Optional LLM reasoning layer: set `ANTHROPIC_API_KEY` (model via `LLM_MODEL`). Without a key, agents use rule-based narratives.
Test headless: `python run_headless.py`
Real data: replace `make_data()` with CSV loaders (sales: week, sku, units; inventory; suppliers).
LangGraph mapping: each `Agent.run(state)` is a node; edges forecast -> working_capital -> procurement -> synthesis.
