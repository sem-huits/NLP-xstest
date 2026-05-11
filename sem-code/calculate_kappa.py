"""
kappa.py  —  Cohen's Kappa between human annotators and LLM
Compares final_label (human) vs. sentiment (LLM) for both
standard and think conditions.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ============================================================================
# LOAD FILES
# ============================================================================
jelle_standard = pd.read_csv('human_annotated/Age-Based Bias in Large Reasoning Models (Jelle).csv')
jelle_think    = pd.read_csv('human_annotated/Age-Based Bias in Large Reasoning Models<think>(J).csv')
rabbi_standard = pd.read_csv('human_annotated/Label-studio-rabbi-Age-Based Bias in Large Reasoning Models-standard.csv')
rabbi_think    = pd.read_csv('human_annotated/Label-studio-rabbi-Age-Based Bias in Large Reasoning Models-think.csv')

# ============================================================================
# HELPER
# ============================================================================
def compute_kappa(df, human_col="final_label", llm_col="sentiment", label=""):
    """Drop rows where either annotation is missing, then compute Cohen's kappa."""
    clean = df[[human_col, llm_col]].dropna()
    clean = clean[(clean[human_col] != "") & (clean[llm_col] != "")]

    if len(clean) == 0:
        print(f"{label}: no overlapping annotations found.")
        return None

    kappa = cohen_kappa_score(clean[human_col], clean[llm_col])
    print(f"{label}")
    print(f"  n = {len(clean)}")
    print(f"  Cohen's κ = {kappa:.4f}")
    print(f"  Label distribution (human):")
    print(clean[human_col].value_counts().to_string(index=True))
    print(f"  Label distribution (LLM):")
    print(clean[llm_col].value_counts().to_string(index=True))
    print()
    return kappa

# ============================================================================
# COMPUTE KAPPA
# ============================================================================
print("=" * 55)
print("Cohen's Kappa: Human vs. LLM")
print("=" * 55)
print()

k1 = compute_kappa(jelle_standard, label="Jelle  — Standard (completion)")
k2 = compute_kappa(rabbi_standard, label="Rabbi  — Standard (completion)")
k3 = compute_kappa(jelle_think,    label="Jelle  — Think (reasoning trace)")
k4 = compute_kappa(rabbi_think,    label="Rabbi  — Think (reasoning trace)")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
results = pd.DataFrame({
    "Annotator": ["Jelle", "Rabbi", "Jelle", "Rabbi"],
    "Condition": ["Standard", "Standard", "Think", "Think"],
    "Cohen's κ": [k1, k2, k3, k4],
})

print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(results.to_string(index=False))

# Interpretation guide
print("""
Interpretation (Landis & Koch, 1977):
  κ < 0.00  →  Poor
  0.00–0.20 →  Slight
  0.21–0.40 →  Fair
  0.41–0.60 →  Moderate
  0.61–0.80 →  Substantial
  0.81–1.00 →  Almost perfect
""")
