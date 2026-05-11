import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

# ============================================================================
# FONT SETUP
# ============================================================================
rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
rc('text', usetex=False)
plt.rcParams.update({'font.size': 20})

# ============================================================================
# LOAD DATA
# ============================================================================
df_dict = {}

for model in ["all"]:
    df_dict[model] = pd.read_csv(f"{model}_results_rejudged.csv")

df = df_dict['all']

# ============================================================================
# OVERVIEW
# ============================================================================
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"\nALL — {len(df)} rows")
print(df['annotation_new'].value_counts().to_string())

# ============================================================================
# BREAKDOWN PER PROMPT TYPE
# ============================================================================
print("\n" + "=" * 50)
print("BREAKDOWN PER PROMPT TYPE")
print("=" * 50)

out_df = (
    df.groupby("type", sort=False)["annotation_new"]
    .value_counts()
    .unstack(fill_value=0)
)
out_df = out_df.div(out_df.sum(axis=1), axis=0)
out_df = out_df.reindex(df["type"].unique().tolist())
print(out_df.map(lambda x: f"{x:.0%}").to_string())

# ============================================================================
# BAR CHART FUNCTION
# ============================================================================
COLORS = {
    '3_full_refusal':       '#ffbbbb',
    '2_partial_compliance': '#ffdf9b',
    '1_full_compliance':    '#90ee90',
}

CLEAN_NAME = {
    'neutral': 'Neutral',
    'child':   'Child (10y)',
    'adult':   'Adult (35y)',
    'elderly': 'Elderly (75y)',
}

AGE_ORDER = ['elderly', 'adult', 'child', 'neutral']

def make_bar_chart(df, plot_title, disable_yticks=False):
    graph_df = pd.DataFrame()

    for age in AGE_ORDER:
        subset = df[df['age_condition'] == age]
        row = (
            subset['annotation_new']
            .value_counts() / len(subset) * 100
        )
        row.name = CLEAN_NAME[age]
        graph_df = pd.concat([graph_df, row.to_frame().T])

    graph_df = graph_df.fillna(0)

    for col in COLORS:
        if col not in graph_df.columns:
            graph_df[col] = 0

    graph_df = graph_df[['3_full_refusal', '2_partial_compliance', '1_full_compliance']]

    ax = graph_df.plot.barh(
        stacked=True,
        figsize=(9, 4),
        color=[COLORS[c] for c in graph_df.columns],
        width=0.6
    )

    plt.tight_layout()
    plt.title(plot_title, y=1.05)
    plt.xlim(0, 100)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend().remove()

    if disable_yticks:
        plt.yticks([])

    return graph_df

# ============================================================================
# PLOT — ALL PROMPTS
# ============================================================================
all_graph_df = make_bar_chart(df, "All Prompts")
plt.savefig('all_prompts_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# REASONING ANALYSIS — MEAN TABLE
# ============================================================================
print("\n" + "=" * 50)
print("MEAN SCORES PER AGE CONDITION")
print("=" * 50)

mean_table = (
    df.groupby("age_condition")[
        ['safety_keyword_density_think', 'safety_keyword_density_resp', 'reasoning_response_gap']
    ]
    .mean()
    .reindex(AGE_ORDER)
    .rename(index=CLEAN_NAME)
)
print(mean_table.round(4).to_string())

# ============================================================================
# REASONING ANALYSIS — BOXPLOTS
# ============================================================================
BOX_PALETTE = ['#ffbbbb', '#ffdf9b', '#90ee90', '#aec6cf']  # elderly, adult, child, neutral

def make_boxplot(metric, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    data_by_age = [df[df['age_condition'] == age][metric].dropna().values for age in AGE_ORDER]
    labels = [CLEAN_NAME[age] for age in AGE_ORDER]

    bp = ax.boxplot(data_by_age, patch_artist=True, widths=0.5)

    for patch, color in zip(bp['boxes'], BOX_PALETTE):
        patch.set_facecolor(color)

    ax.set_xticklabels(labels)
    ax.set_title(title, y=1.02)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

make_boxplot(
    'safety_keyword_density_think',
    'Safety Keyword Density — Reasoning Trace',
    'Density',
    'boxplot_density_think.png'
)

make_boxplot(
    'safety_keyword_density_resp',
    'Safety Keyword Density — Response',
    'Density',
    'boxplot_density_resp.png'
)

make_boxplot(
    'reasoning_response_gap',
    'Reasoning–Response Safety Gap',
    'Gap',
    'boxplot_gap.png'
)

# ============================================================================
# STATISTICAL TESTS — KRUSKAL-WALLIS + MANN-WHITNEY U
# ============================================================================
print("\n" + "=" * 50)
print("STATISTICAL TESTS")
print("=" * 50)

metrics = {
    'safety_keyword_density_think': 'Safety Keyword Density (Think)',
    'safety_keyword_density_resp':  'Safety Keyword Density (Response)',
    'reasoning_response_gap':       'Reasoning–Response Gap',
}

for col, label in metrics.items():
    print(f"\n--- {label} ---")

    groups = [df[df['age_condition'] == age][col].dropna().values for age in AGE_ORDER]

    stat, p = kruskal(*groups)
    print(f"Kruskal-Wallis: H = {stat:.4f}, p = {p:.4f}")

    if p < 0.05:
        print("Significant → pairwise Mann-Whitney U (Bonferroni corrected):")
        pairs = list(combinations(AGE_ORDER, 2))
        alpha_corrected = 0.05 / len(pairs)

        for a, b in pairs:
            g1 = df[df['age_condition'] == a][col].dropna().values
            g2 = df[df['age_condition'] == b][col].dropna().values
            u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
            sig = "✓" if p_val < alpha_corrected else " "
            print(f"  [{sig}] {CLEAN_NAME[a]} vs {CLEAN_NAME[b]}: U = {u_stat:.1f}, p = {p_val:.4f} (threshold: {alpha_corrected:.4f})")
    else:
        print("Not significant — no pairwise tests performed.")

print("\nCharts saved.")
