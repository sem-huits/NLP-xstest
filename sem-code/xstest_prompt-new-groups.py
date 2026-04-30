import pandas as pd

# Configuration
INPUT_CSV = '/Users/semhuits/Library/CloudStorage/OneDrive-UniversiteitLeiden/Master Data Science leiden/Natural Language Processing for Social Good/Final-assignment/xstest_prompts.csv'  # Change this to your CSV file path
PROMPT_COLUMN = 'prompt'  # Change if your column has a different name

# Age groups with their prefixes
age_groups = {
    'Child': 'I am 10 years old. ',
    'Adult': 'I am 35 years old. ',
    'Elderly': 'I am 75 years old. '
}

# Read original CSV
df_original = pd.read_csv(INPUT_CSV)

# Generate and save each age-prefixed CSV
for age_name, prefix in age_groups.items():
    # Create new dataframe with prefixed prompts
    df_new = df_original.copy()
    df_new[PROMPT_COLUMN] = prefix + df_new[PROMPT_COLUMN].astype(str)
    
    # Save to new CSV
    output_filename = f'{age_name.lower()}_prompts.csv'
    df_new.to_csv(output_filename, index=False)
    
    print(f"✓ Created {output_filename}")
    print(f"  Rows: {len(df_new)}")
    print(f"  Sample: {df_new[PROMPT_COLUMN].iloc[0]}\n")

print("All files created successfully!")