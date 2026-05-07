import pandas as pd

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv('all_results.csv')

print(f"Totaal rijen: {len(df)}")
print(df['age_condition'].value_counts())

# ============================================================================
# STRATIFIED SAMPLE — 25 per age condition = 100 total
# Zelfde sample voor beide files (zelfde random_state!)
# ============================================================================
sample = (
    df.groupby('age_condition', group_keys=False)
    .apply(lambda x: x.sample(n=min(25, len(x)), random_state=42))
    .reset_index(drop=True)
)

# ============================================================================
# EXPORT 1 — completion (eindantwoord)
# ============================================================================
cols_completion = ['id', 'type', 'age_condition', 'prompt', 'completion', 'final_label']
cols_completion = [c for c in cols_completion if c in sample.columns]

sample[cols_completion].to_csv('kappa_sample_completion.csv', index=False)
print(f"\nkappa_sample_completion.csv opgeslagen: {len(sample)} rijen")

# ============================================================================
# EXPORT 2 — reasoning_trace (think gedeelte)
# ============================================================================
cols_think = ['id', 'type', 'age_condition', 'prompt', 'reasoning_trace', 'final_label']
cols_think = [c for c in cols_think if c in sample.columns]

sample[cols_think].to_csv('kappa_sample_think.csv', index=False)
print(f"kappa_sample_think.csv opgeslagen: {len(sample)} rijen")

print("\nAge condition verdeling:")
print(sample['age_condition'].value_counts())