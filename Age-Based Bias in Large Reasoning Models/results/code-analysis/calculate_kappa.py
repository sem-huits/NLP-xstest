import pandas as pd
from sklearn.metrics import cohen_kappa_score

# === LAAD BESTANDEN ===
rejudged = pd.read_csv("all_results_rejudged.csv")

rabbi_std   = pd.read_csv("human_annotated/Label-studio-rabbi-Age-Based Bias in Large Reasoning Models-standard.csv")
jelle_std   = pd.read_csv("human_annotated/Age-Based Bias in Large Reasoning Models (Jelle).csv")
rabbi_think = pd.read_csv("human_annotated/Label-studio-rabbi-Age-Based Bias in Large Reasoning Models-think.csv")
jelle_think = pd.read_csv("human_annotated/Age-Based Bias in Large Reasoning Models<think>(J).csv")

# === SPLITS REJUDGED OP TYPE ===
# Standard = heeft completion maar geen reasoning_trace
# Think    = heeft reasoning_trace
rejudged_std   = rejudged[rejudged['reasoning_trace'].isna()][['id', 'age_condition', 'annotation_new']].copy()
rejudged_think = rejudged[rejudged['reasoning_trace'].notna()][['id', 'age_condition', 'annotation_new']].copy()

KEY = ['id', 'age_condition']

def calc_kappa(rejudged_subset, human_df, label_human, label_name):
    merged = rejudged_subset.merge(
        human_df[KEY + ['final_label']],
        on=KEY
    )
    missing = len(rejudged_subset) - len(merged)
    if missing > 0:
        print(f"  ⚠️  {missing} rijen niet gematcht voor {label_name}")

    kappa = cohen_kappa_score(merged['annotation_new'], merged['final_label'])
    agree = (merged['annotation_new'] == merged['final_label']).mean()
    print(f"  {label_name}")
    print(f"    Gematcht:    {len(merged)}")
    print(f"    Agreement:   {agree:.1%}")
    print(f"    Kappa:       {kappa:.4f}")
    return merged

print("=== STANDARD ===")
m1 = calc_kappa(rejudged_std, rabbi_std,   'final_label', 'Rejudged vs Rabbi')
m2 = calc_kappa(rejudged_std, jelle_std,   'final_label', 'Rejudged vs Jelle')

print("\n=== THINK ===")
m3 = calc_kappa(rejudged_think, rabbi_think, 'final_label', 'Rejudged vs Rabbi')
m4 = calc_kappa(rejudged_think, jelle_think, 'final_label', 'Rejudged vs Jelle')
