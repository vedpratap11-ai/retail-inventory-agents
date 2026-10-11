from data_gen import make_data, DEFAULT_CONFIG
from agents import Orchestrator
S = Orchestrator().run(make_data(), DEFAULT_CONFIG)
print(S["plan"]); print(S["log"]); print(S["pos"].head(8).to_string()); print(len(S["exceptions"]), "exceptions")
