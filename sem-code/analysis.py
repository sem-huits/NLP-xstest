# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

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

for model in ["neutral", "child", "adult", "elderly"]:
    df_dict[model] = pd.read_csv(f"{model}_results.csv")

# ============================================================================
# OVERVIEW
# ============================================================================
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
for m in df_dict:
    df = df_dict[m]
    print(f"\n{m.upper()} — {len(df)} rows")
    print(df['final_label'].value_counts().to_string())

# ============================================================================
# BREAKDOWN PER PROMPT TYPE
# ============================================================================
print("\n" + "=" * 50)
print("BREAKDOWN PER PROMPT TYPE")
print("=" * 50)

for m in df_dict:
    print("\n" + "#" * 40)
    print(f" {m.upper()}")
    print("#" * 40)

    out_df = (
        df_dict[m]
        .groupby("type", sort=False)["final_label"]
        .value_counts()
        .unstack(fill_value=0)
    )
    out_df = out_df.div(out_df.sum(axis=1), axis=0)
    out_df = out_df.reindex(df_dict[m]["type"].unique().tolist())

    # Format as percentages for printing
    print(out_df.map(lambda x: f"{x:.0%}").to_string())

# ============================================================================
# BREAKDOWN SAFE VS UNSAFE (contrast)
# ============================================================================
print("\n" + "=" * 50)
print("BREAKDOWN SAFE VS UNSAFE")
print("=" * 50)

for m in df_dict:
    print("\n" + "#" * 40)
    print(f" {m.upper()}")
    print("#" * 40)

    out_df = (
        df_dict[m]
        .groupby(df_dict[m]["type"].str.contains("contrast"), sort=False)["final_label"]
        .value_counts()
        .unstack(fill_value=0)
    )
    out_df.index = ["Safe", "Unsafe"]
    out_df = out_df.div(out_df.sum(axis=1), axis=0)
    print(out_df.map(lambda x: f"{x:.1%}").to_string())

# ============================================================================
# BAR CHART FUNCTION
# ============================================================================
COLORS = {
    '3_full_refusal':     '#ffbbbb',
    '2_partial_compliance': '#ffdf9b',
    '1_full_compliance':  '#90ee90',
}

CLEAN_NAME = {
    'neutral': 'Neutral',
    'child':   'Child (10y)',
    'adult':   'Adult (35y)',
    'elderly': 'Elderly (75y)',
}

def make_bar_chart(data_dict, plot_title, disable_yticks=False):
    graph_df = pd.DataFrame()

    for m in ['elderly', 'adult', 'child', 'neutral']:  # reversed for horizontal bar
        row = (
            data_dict[m]['final_label']
            .value_counts() / len(data_dict[m]) * 100
        )
        row.name = CLEAN_NAME[m]
        graph_df = pd.concat([graph_df, row.to_frame().T])

    graph_df = graph_df.fillna(0)

    # Ensure all columns exist
    for col in COLORS:
        if col not in graph_df.columns:
            graph_df[col] = 0

    # Sort columns by severity
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
# SPLIT SAFE / UNSAFE
# ============================================================================
safe_dict   = {m: df_dict[m][~df_dict[m]['type'].str.contains('contrast')].copy() for m in df_dict}
unsafe_dict = {m: df_dict[m][df_dict[m]['type'].str.contains('contrast')].copy()  for m in df_dict}

# ============================================================================
# PLOT — SAFE PROMPTS
# ============================================================================
safe_graph_df = make_bar_chart(safe_dict, "Safe Prompts", disable_yticks=False)
plt.savefig('safe_prompts_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT — UNSAFE PROMPTS
# ============================================================================
unsafe_graph_df = make_bar_chart(unsafe_dict, "Contrasting Unsafe Prompts", disable_yticks=False)
plt.savefig('unsafe_prompts_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT — ALL PROMPTS COMBINED
# ============================================================================
all_graph_df = make_bar_chart(df_dict, "All Prompts", disable_yticks=False)
plt.savefig('all_prompts_chart.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCharts saved: safe_prompts_chart.png, unsafe_prompts_chart.png, all_prompts_chart.png")
