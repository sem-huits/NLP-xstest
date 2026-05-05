import pandas as pd

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv('all_results.csv')

print(f"Totaal rijen: {len(df)}")
print(df['age_condition'].value_counts())

# ============================================================================
# STRATIFIED SAMPLE — 25 per age condition = 100 total
# ============================================================================
sample = (
    df.groupby('age_condition', group_keys=False)
    .apply(lambda x: x.sample(n=25, random_state=42))
    .reset_index(drop=True)
)

# ============================================================================
# EXPORT — only columns needed for manual annotation
# ============================================================================
cols = ['id', 'type', 'age_condition', 'prompt', 'completion', 'final_label']
cols = [c for c in cols if c in sample.columns]  # only existing columns

sample[cols].to_csv('kappa_sample.csv', index=False)

print(f"\nkappa_sample.csv opgeslagen: {len(sample)} rijen")
print(sample['age_condition'].value_counts())
