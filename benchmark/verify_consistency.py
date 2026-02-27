"""Verify cross-table Sharpe consistency."""
import json

with open("reports/all_metrics.json") as f:
    t1 = json.load(f)
with open("reports/robustness_metrics.json") as f:
    rob = json.load(f)
with open("reports/adversarial_defense_metrics.json") as f:
    adv = json.load(f)

print("=== CROSS-TABLE CLEAN SHARPE COMPARISON ===")
for name in ["MLP", "LSTM"]:
    s1 = round(t1["main"][name]["Sharpe (gross)"], 3)
    s9 = round(rob["adversarial"][name]["0.01"]["sharpe_clean"], 3)
    s13 = round(list(adv[name + "_Standard"].values())[0]["sharpe_clean"], 3)
    match = "MATCH" if s1 == s9 == s13 else "MISMATCH"
    print(f"  {name}: Table1={s1}  Table9={s9}  Table13={s13}  -> {match}")
