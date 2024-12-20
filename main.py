import os
import numpy as np
print(np.__version__)
# np 1.26.6
import pandas as pd
import plotly.graph_objects as go
import openpyxl
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, SVG
from PIL import Image
import statsmodels.formula.api as smf
import re

import stata_setup
stata_setup.config("/Applications/Stata", "mp")

import ipystata
from pystata import stata

# Set up the working directory using the state abbreviation
work_dir = '/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest'

# Change the working directory
os.chdir(work_dir)  # Changes the current working directory to the specified path, ensuring that all subsequent file operations are relative to this directory.

# Get the current working directory
cwd = os.getcwd()  # Retrieves the current working directory path after the change.

# Print the current working directory
print("Current working directory: {0}".format(cwd))  # Outputs the current working directory path to confirm that the change was successful.

raw_path = './data/raw/'
processed_path = './data/processed/'

#%%

df = pd.read_csv('data/raw/county_land_use.csv')

socioeconomic_df = pd.read_excel('data/raw/cntypanel_2000_2020.xlsx')

df = pd.merge(df,socioeconomic_df, on=['cntyid','year'], suffixes=('', '_drop'), how='left')

df = df.loc[:, ~df.columns.str.endswith('_drop')]

df = df.drop([col for col in df.columns if col.endswith('.1')], axis=1)

df['population'] = df.apply(lambda row: row['population'] * 10000 if row['year'] <= 2017 else row['population'], axis=1)

poverty_df = pd.read_excel('./data/raw/poverty_county_info.xlsx')

df = pd.merge(poverty_df, df, on=['cntyid'], suffixes=('_drop', ''), how='outer')

# Drop columns with '_drop' suffix
df = df.loc[:, ~df.columns.str.endswith('_drop')]

county_info_df = pd.read_excel('./data/raw/county_basic_info.xlsx')

df = pd.merge(county_info_df, df, on=['cntyid'], suffixes=('_drop',''), how='outer')

df = df.drop_duplicates()

county_info_df = pd.read_excel('./data/raw/county_basic_info.xlsx')

df = pd.merge(county_info_df, df, on=['cntyid'], suffixes=('_drop',''), how='outer')

rdls_df = pd.read_excel(raw_path + 'rdls.xlsx')
df = pd.merge(df, rdls_df, on=['cntyid'], suffixes=('','_drop'), how='left')


lightdata_df = pd.read_excel(raw_path + 'lightdata.xlsx')
df = pd.merge(df, lightdata_df, on=['cntyid', 'year'], suffixes=('','_drop'), how='left')


rainfall_df = pd.read_excel(raw_path + 'rainfall.xlsx')
df = pd.merge(df, rainfall_df, on=['cntyid', 'year'], suffixes=('','_drop'), how='left')
df['rainfall_square'] = df['rainfall'] ** 2

windspeed_df = pd.read_excel(raw_path + 'windspeed.xlsx')
df = pd.merge(df, windspeed_df, on=['cntyid', 'year'], suffixes=('','_drop'), how='left')
df['wind_speed_square'] = df['wind_speed'] ** 2

forest_clean_df = pd.read_excel(raw_path + 'forest_clean.xlsx')
# List of columns to convert to numeric
columns_to_convert = ['manual_afforestation', 'aerial_afforestation', 'natural_reforestation',
                      'forest_recovering', 'manual_reforestation']
# Loop through each column and apply pd.to_numeric with error handling
for column in columns_to_convert:
    forest_clean_df[column] = pd.to_numeric(forest_clean_df[column], errors='coerce')
forest_clean_df = forest_clean_df.groupby(['cntyid', 'year']).sum().reset_index()
df = pd.merge(df, forest_clean_df, on=['cntyid', 'year'], suffixes=('','_drop'), how='left')

df['reforestation'] = (
    df['manual_afforestation'] +
    df['aerial_afforestation'] +
    df['natural_reforestation'] +
    df['forest_recovering'] +
    df['manual_reforestation']
) / 100

# Use a list comprehension with two ranges (1-9 for both X and Y)
land_use_change_columns = [f'land_use_{x}{y}' for x in range(1, 10) for y in range(1, 10)]

for col in land_use_change_columns:
    new_col = f'share_{col}'  # Creating the new column name, e.g., 'share_land_use_12'
    df[new_col] = df[col] / df['total_area']  # Calculating the share and assigning it to the new column

# List of columns that start with 'land_use_'
land_use_columns = [col for col in df.columns if re.match(r'land_use_[1-9]$', col)]

# Generate the ratio of each land use area to total_area_km2
for col in land_use_columns:
    df[f'share_{col}'] = df[col] / df['total_area']

village_df = pd.read_csv(raw_path + 'county_village_numbers.csv')
df = pd.merge(df, village_df , on=['cntyid', 'year'], suffixes=('','_drop'), how='left')
df = df.loc[:, ~df.columns.str.endswith('_drop')]

north_county_df = pd.read_csv(raw_path + 'north_county.csv')
df = pd.merge(df, north_county_df, on='cntyid', suffixes=('','_drop'), how='left')
df = df.loc[:, ~df.columns.str.endswith('_drop')]

carbon_storage_df = pd.read_csv(raw_path + 'carbon_storage.csv')
df = pd.merge(df, carbon_storage_df, on=['cntyid', 'year'], suffixes=('', '_drop'), how='left')

ndvi_df = pd.read_csv(raw_path + 'ndvi.csv')
df = pd.merge(df, ndvi_df, on=['cntyid', 'year'], suffixes=('', '_drop'), how='left')

df = df.loc[:, ~df.columns.str.endswith('_drop')]
df = df.drop_duplicates()
# Save the DataFrame to a Stata .dta file
df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
df = df.loc[:, ~df.columns.str.endswith('.1')]

forest_gain_columns = [f'land_use_{i}' for i in range(12, 100, 10) if i != 22]

df['forest_gain'] = df[forest_gain_columns].sum(axis=1)

df['share_forest_gain'] = df['forest_gain']/df['total_area']

forest_loss_columns = [f'land_use_{i}' for i in range(21, 30, 1) if i != 22]

df['forest_loss'] = df[forest_loss_columns].sum(axis=1)

df['share_forest_loss'] = df['forest_loss']/df['total_area']

df['net_share_forest_gain'] = df['share_forest_gain'] - df['share_forest_loss']

df.to_csv(processed_path + '/main.csv', encoding='utf-8')

df = pd.read_csv(processed_path + '/main.csv', encoding='utf-8')

# cols_to_move = ['share_forest_gain', 'forest_gain','share_forest_loss', 'forest_loss','land_use_2',
#                 'share_land_use_2', 'net_share_forest_gain','land_use_22','share_land_use_22','land_use_12',
#                 'share_land_use_12', 'total_area',
#                 'land_use_sum']
#
# # Reorder the DataFrame by moving the specified columns to the front
# df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
#
#
# exclude_cols = [f'land_use_{i}' for i in range(20, 100, 10)]
# cols_to_sum = [f'land_use_{i}' for i in range(11, 100) if f'land_use_{i}' in df.columns and f'land_use_{i}' not in exclude_cols]
#
# # Calculate the sum of the selected columns
# df['land_use_sum'] = df[cols_to_sum].sum(axis=1)


#%% data selection main results

df = df[(df['minority_county'] != 1) & (df['key_poverty_county'] != 1)]

df = df[df['year']>= 2000]

df = df[pd.notna(df['county'])] # only in statis yearbook

#df = df[df['county'].notna()]

for i in df.columns:
    print(i)

# Create or modify 'treatment_group' column using .loc
df.loc[df['new_poor_cnty'] == 1, 'treated'] = 1

df.loc[pd.isnull(df['poor_cnty']), 'treated'] = 0

print(df['treated'].sum())
df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
df.to_csv(processed_path + '/main_short.csv', encoding='utf-8')

#%% summary
df = pd.read_csv(processed_path + '/main_short.csv', encoding='utf-8')

df['share_land_use_2_100'] = df['share_land_use_2'] * 100
df['share_forest_gain_100'] = df['share_forest_gain'] * 100
df['share_forest_loss_100'] = df['share_forest_loss'] * 100
df['net_share_forest_gain_100'] = df['net_share_forest_gain'] * 100
df['share_reforestation_100'] = df['reforestation']/df['total_area'] * 100
df['population_1000'] = df['population'] / 1000
df['carbon_density'] = df['carbon_value'] / df['total_area']
df['NDVI'] = df['NDVI_mean']/10000
df['wind_speed_mph'] = df['wind_speed'] * 2.23694  # Convert to mph
exchange_rate = 6.5250
df['vad_pri_million_usd'] = df['vad_pri'] / 100 / exchange_rate
df['vad_sec_million_usd'] = df['vad_sec'] / 100 / exchange_rate
df['gov_rev_million_usd'] = df['gov_rev'] / 100 / exchange_rate
df['gov_exp_million_usd'] = df['gov_exp'] / 100 / exchange_rate
df['savings_million_usd'] = df['savings'] / 100 / exchange_rate

for i in range(1, 10):
    column_name = f'share_land_use_{i}'
    new_column_name = f'{column_name}_100'
    df[new_column_name] = df[column_name] * 100

variable_labels = {
    'share_land_use_2_100': 'Forest Share(\\%)',
    'share_forest_gain_100': 'Forest Gains per $km^2$(\\%)',
    'share_forest_loss_100': 'Forest Loss per $km^2$ (\\%)',
    'net_share_forest_gain_100': 'Forest Share Change (\\%)',
    'share_reforestation_100': 'Planted Forest Share (\\%)',
    'share_land_use_1_100': 'Cropland Share(\\%)',
    'share_land_use_3_100': 'Shrub Share(\\%)',
    'share_land_use_4_100': 'Grassland Share(\\%)',
    'share_land_use_5_100': 'Water Share(\\%)',
    'share_land_use_6_100': 'Snow Share(\\%)',
    'share_land_use_7_100': 'Barren Land Share(\\%)',
    'share_land_use_8_100': 'Impervious Surface Share(\\%)',
    'share_land_use_9_100': 'Wetland Share(\\%)',
    'carbon_density': 'Carbon Storage Density ($C$ ton/$km^2$)',
    'NDVI': 'Average NDVI per $km^2$',
    'total_area': 'County Area ($km^2$)',
    'population_1000': 'Population (Thousand)',
    'gov_rev_million_usd': "Gov't Revenue (Million USD)",
    'gov_exp_million_usd': "Gov't Expenditure (Million USD)",
    'vad_pri_million_usd': 'GDP Primary (Million USD)',
    'vad_sec_million_usd': 'GDP Secondary (Million USD)',
    'ur_code_220': 'Number of Rural Villages',
    'rdls': 'Relief Degree of Land Surface',
    'savings_million_usd': 'Savings Deposit (Million USD)',
    'lightmean': 'Average NTL Intensity per $km^2$',
    'rainfall': 'Average Annual Precipitation (mm)',
    'wind_speed_mph': 'Average Annual Wind Speed (mph)'
}

selected_vars = list(variable_labels.keys())


# Group by 'treated' and calculate mean and SD
grouped_stats = df.groupby('treated')[selected_vars].agg(['mean', 'std'])

# Flatten the MultiIndex columns
grouped_stats.columns = ['_'.join(col) for col in grouped_stats.columns]

# Reshape the DataFrame to prepare for LaTeX output
summary_stats = pd.DataFrame({
    'Variable': [variable_labels[var] for var in selected_vars],
    'Treated Mean': grouped_stats.loc[1, [f'{var}_mean' for var in selected_vars]].values,
    'Treated SD': grouped_stats.loc[1, [f'{var}_std' for var in selected_vars]].values,
    'Untreated Mean': grouped_stats.loc[0, [f'{var}_mean' for var in selected_vars]].values,
    'Untreated SD': grouped_stats.loc[0, [f'{var}_std' for var in selected_vars]].values
})

treated_obs = int(df[df['treated'] == 1].shape[0])
untreated_obs = int(df[df['treated'] == 0].shape[0])
total_obs = len(df)

treated_unique_cntyid = int(df[df['treated'] == 1]['cntyid'].nunique())
untreated_unique_cntyid = int(df[df['treated'] == 0]['cntyid'].nunique())

# Add total observations row
obs_row = pd.DataFrame({
    'Variable': ['Total Observations'],
    'Treated Mean': [treated_obs],
    'Treated SD': [''],
    'Untreated Mean': [untreated_obs],
    'Untreated SD': [''],
})

cntyid_row = pd.DataFrame({
    'Variable': ['No. of Counties'],
    'Treated Mean': [treated_unique_cntyid],
    'Treated SD': [''],
    'Untreated Mean': [untreated_unique_cntyid],
    'Untreated SD': [''],
})

final_summary_stats = pd.concat([summary_stats, obs_row,cntyid_row], ignore_index=True)

# Generate LaTeX table
latex_table = final_summary_stats.to_latex(
    index=False,
    caption='Descriptive Statistics',
    label='t:descriptive_statistics',
    column_format='lrrrr',
    float_format="%.2f",
    header=['Variable', 'Mean', 'SD', 'Mean', 'SD'],
    na_rep=''
)

# Add a new row before the header for "Treated" and "Untreated"
additional_row = r'\multicolumn{1}{c}{} & \multicolumn{2}{c}{Treated} & \multicolumn{2}{c}{Untreated} \\'
additional_row += '\n'

# Split the LaTeX table into lines for manipulation
latex_table_lines = latex_table.splitlines()

# Insert the additional row before the main header (after the \toprule line)
for i, line in enumerate(latex_table_lines):
    if 'toprule' in line:
        latex_table_lines.insert(i + 1, additional_row)
        break

# Join the lines back into a single string
latex_table = '\n'.join(latex_table_lines)

# Convert integer values to integer format in LaTeX
latex_table = latex_table.replace(
    f'{treated_obs}.00', str(treated_obs)
).replace(
    f'{untreated_obs}.00', str(untreated_obs)
).replace(
    f'{treated_unique_cntyid}.00', str(treated_unique_cntyid)
).replace(
    f'{untreated_unique_cntyid}.00', str(untreated_unique_cntyid)
)

# # Add \resizebox to make the table text-wide
# latex_table = (
#     "\\begin{table}[!htbp]\n"
#     "\\centering\n"
#     "\\caption{Descriptive Statistics}\n"
#     "\\label{t:descriptive_statistics}\n"
#     "\\resizebox{\\textwidth}{!}{%\n"
#     + latex_table +
#     "}\n\\end{table}"
# )

# Print the LaTeX code
print(latex_table)

# Save the LaTeX table to a .tex file
file_path = './code/writing/tables/direct_summary_statistics.tex'
with open(file_path, 'w') as file:
    file.write(latex_table)


#%% summarize other land uses


exclude_columns = {20, 22, 30, 33, 40, 44, 50, 55, 60, 66, 70, 77, 80, 88, 90}

# Loop through columns from 12 to 98
for i in range(12, 99):
    if i not in exclude_columns:  # Check if the column index is not in the exclusion list
        column_name = f'share_land_use_{i}'
        new_column_name = f'{column_name}_100'
        df[new_column_name] = df[column_name] * 100

land_use_mapping = {
    1: 'Cropland',
    2: 'Forest',
    3: 'Shrub',
    4: 'Grassland',
    5: 'Water',
    6: 'Snow',
    7: 'Barren',
    8: 'Impervious',
    9: 'Wetland'
}

# Exclusion list for certain columns
exclude_columns = {20, 22, 30, 33, 40, 44, 50, 55, 60, 66, 70, 77, 80, 88, 90}

# Dictionary to store variable labels
long_variable_labels = {}

# Loop through columns from 12 to 98 (excluding specified ones)
for i in range(12, 99):
    if i not in exclude_columns:
        # Extract source and target land use types based on the first and second digits of the column index
        source_land_use = land_use_mapping.get(i // 10, 'Unknown')
        target_land_use = land_use_mapping.get(i % 10, 'Unknown')

        # Create the label for the column
        label = f"{source_land_use} to {target_land_use} (\\%)"

        # Add the column label to the dictionary
        column_name = f'share_land_use_{i}_100'
        long_variable_labels[column_name] = label

selected_vars = list(long_variable_labels.keys())


# Group by 'treated' and calculate mean and SD
grouped_stats = df.groupby('treated')[selected_vars].agg(['mean', 'std'])

# Flatten the MultiIndex columns
grouped_stats.columns = ['_'.join(col) for col in grouped_stats.columns]

# Reshape the DataFrame to prepare for LaTeX output
summary_stats = pd.DataFrame({
    'Variable': [long_variable_labels[var] for var in selected_vars],
    'Treated Mean': grouped_stats.loc[1, [f'{var}_mean' for var in selected_vars]].values,
    'Treated SD': grouped_stats.loc[1, [f'{var}_std' for var in selected_vars]].values,
    'Untreated Mean': grouped_stats.loc[0, [f'{var}_mean' for var in selected_vars]].values,
    'Untreated SD': grouped_stats.loc[0, [f'{var}_std' for var in selected_vars]].values
})

treated_obs = int(df[df['treated'] == 1].shape[0])
untreated_obs = int(df[df['treated'] == 0].shape[0])
total_obs = len(df)

treated_unique_cntyid = int(df[df['treated'] == 1]['cntyid'].nunique())
untreated_unique_cntyid = int(df[df['treated'] == 0]['cntyid'].nunique())

final_summary_stats = pd.concat([summary_stats], ignore_index=True)

# Generate LaTeX table
latex_table = final_summary_stats.to_latex(
    index=False,
    caption='Descriptive Statistics',
    label='t:descriptive_statistics',
    column_format='lrrrr',
    float_format="%.2f",
    header=['Variable', 'Mean', 'SD', 'Mean', 'SD'],
    na_rep=''
)


# Print the LaTeX code
print(latex_table)

# Save the LaTeX table to a .tex file
file_path = './code/writing/tables/direct_long_summary_statistics.tex'
with open(file_path, 'w') as file:
    file.write(latex_table)



#%% mountain area summary


# mountain area summary
# Group by 'mountain_area' and count unique 'cntyid' in each group
mountain_summary = df.groupby('mountain_area_en')['cntyid'].nunique().reset_index()

# Rename columns for clarity
mountain_summary.columns = ['Mountain Region', 'No. of Poverty County in Study']

# Save to LaTeX
latex_code = mountain_summary.to_latex(index=False)

# Generate LaTeX code with a label
latex_code = (
    "\\begin{table}[h!]\n"
    "\\centering\n"
    f"{mountain_summary.to_latex(index=False)}"
    "\\caption{Count of Poverty Counties Across Mountain Regions in Study}\n"
    "\\label{tab:mountain_summary}\n"
    "\\end{table}\n"
)

# Save the LaTeX code into a .tex file
with open('outputs/mountain_summary.tex', 'w') as f:
    f.write(latex_code)

with open('code/writing/tables/mountain_summary.tex', 'w') as f:
    f.write(latex_code)

# more details

translation = {
    '大兴安岭南麓山区': 'Southern Daxing’anling Mountain Area',
    '燕山-太行山区': 'Yanshan-Taihang Mountain Area',
    '六盘山区': 'Liupan Mountain Area',
    '秦巴山区': 'Qinba Mountain Area',
    '大别山区': 'Dabie Mountain Area',
    '乌蒙山区': 'Wumeng Mountain Area',
    '武陵山区': 'Wuling Mountain Area',
    '滇西边境山区': 'Western Yunnan Border Mountain Area',
    '滇桂黔石漠化区': 'Dian-Gui-Qian Karst Region',
    '罗霄山区': 'Luoxiao Mountain Area'
}

df['mountain_area_en'] = df['mountain_area'].map(translation)

df['mountain_area_en'] = df['mountain_area_en'].fillna('Control Group')

print(df['mountain_area_en'].unique())

df['share_land_use_2_100'] = df['share_land_use_2'] * 100
df['share_forest_gain_100'] = df['share_forest_gain'] * 100
df['share_forest_loss_100'] = df['share_forest_loss'] * 100
df['net_share_forest_gain_100'] = df['net_share_forest_gain'] * 100
df['share_reforestation_100'] = df['reforestation']/df['total_area'] * 100
df['population_1000'] = df['population'] / 1000
df['carbon_density'] = df['carbon_value'] / df['total_area']
df['NDVI'] = df['NDVI_mean']/10000
df['wind_speed_mph'] = df['wind_speed'] * 2.23694  # Convert to mph
exchange_rate = 6.5250
df['vad_pri_million_usd'] = df['vad_pri'] / 100 / exchange_rate
df['vad_sec_million_usd'] = df['vad_sec'] / 100 / exchange_rate
df['gov_rev_million_usd'] = df['gov_rev'] / 100 / exchange_rate
df['gov_exp_million_usd'] = df['gov_exp'] / 100 / exchange_rate
df['savings_million_usd'] = df['savings'] / 100 / exchange_rate

for i in range(1, 10):
    column_name = f'share_land_use_{i}'
    new_column_name = f'{column_name}_100'
    df[new_column_name] = df[column_name] * 100

variable_labels = {
    'share_land_use_2_100': 'Forest Share (\\%)',
    'share_forest_gain_100': 'Forest Gains per $km^2$ (\\%)',
    'share_forest_loss_100': 'Forest Loss per $km^2$ (\\%)',
    'net_share_forest_gain_100': 'Forest Share Change (\\%)',
    'share_reforestation_100': 'Planted Forest Share (\\%)',
    'share_land_use_1_100': 'Cropland Share (\\%)',
    'share_land_use_3_100': 'Shrub Share (\\%)',
    'share_land_use_4_100': 'Grassland Share (\\%)',
    'share_land_use_5_100': 'Water Share (\\%)',
    'share_land_use_6_100': 'Snow Share (\\%)',
    'share_land_use_7_100': 'Barren Land Share (\\%)',
    'share_land_use_8_100': 'Impervious Surface Share (\\%)',
    'share_land_use_9_100': 'Wetland Share (\\%)',
    'carbon_density': 'Carbon Storage Density ($C$ ton/$km^2$)',
    'NDVI': 'Average NDVI per $km^2$',
    'total_area': 'County Area ($km^2$)',
    'population_1000': 'Population (Thousand)',
    'gov_rev_million_usd': "Gov't Revenue (Million USD)",
    'gov_exp_million_usd': "Gov't Expenditure (Million USD)",
    'vad_pri_million_usd': 'GDP Primary (Million USD)',
    'vad_sec_million_usd': 'GDP Secondary (Million USD)',
    'ur_code_220': 'Number of Rural Villages',
    'rdls': 'Relief Degree of Land Surface',
    'savings_million_usd': 'Savings Deposit (Million USD)',
    'lightmean': 'Average NTL Intensity per $km^2$',
    'rainfall': 'Average Annual Precipitation (mm)',
    'wind_speed_mph': 'Average Annual Wind Speed (mph)'
}

selected_vars = list(variable_labels.keys())

# Group by 'mountain_area_en' and calculate mean
grouped_stats = df.groupby('mountain_area_en')[selected_vars].agg(['mean'])
grouped_stats.columns = ['_'.join(col) for col in grouped_stats.columns]

# Reorder mountain_areas list
mountain_areas = grouped_stats.index.tolist()
if 'Control Group' in mountain_areas:
    mountain_areas.remove('Control Group')
    mountain_areas.append('Control Group')

# Create the summary_stats DataFrame
summary_stats = pd.DataFrame({'Variable': [variable_labels[var] for var in selected_vars]})
for area in mountain_areas:
    summary_stats[area] = grouped_stats.loc[area, [f'{var}_mean' for var in selected_vars]].values

# Round numeric values to two decimal places
for col in summary_stats.columns[1:]:
    summary_stats[col] = pd.to_numeric(summary_stats[col], errors='coerce').round(2)

# Add observation counts
obs_counts = df['mountain_area_en'].value_counts()
obs_row = {'Variable': 'Observations'}
for area in mountain_areas:
    obs_row[area] = int(obs_counts.get(area, 0))

# Add unique 'cntyid' counts
unique_cntyid_counts = df.groupby('mountain_area_en')['cntyid'].nunique()
unique_cntyid_row = {'Variable': 'No. of Counties'}
for area in mountain_areas:
    unique_cntyid_row[area] = unique_cntyid_counts.get(area, 0)

# Concatenate the DataFrames
obs_row_df = pd.DataFrame([obs_row])
unique_cntyid_row_df = pd.DataFrame([unique_cntyid_row])
# Convert numeric columns to integers
for col in obs_row_df.columns[1:]:
    obs_row_df[col] = pd.to_numeric(obs_row_df[col], errors='coerce').astype(int)

for col in unique_cntyid_row_df.columns[1:]:
    unique_cntyid_row_df[col] = pd.to_numeric(unique_cntyid_row_df[col], errors='coerce').astype(int)

final_summary_stats = pd.concat([summary_stats, obs_row_df, unique_cntyid_row_df], ignore_index=True)
for col in final_summary_stats.columns[1:]:  # Skip the 'Variable' column
    final_summary_stats.loc[final_summary_stats.index[-2:], col] = (
        final_summary_stats.loc[final_summary_stats.index[-2:], col].astype(int)
    )

# Create the LaTeX table
latex_table = r"""
\begin{table}[ht]
\centering
\begin{tabular}{l""" + "c" * len(mountain_areas) + r"""}
\hline
\textbf{Variable} & """ + ' & '.join([f'\\textbf{{{area}}}' for area in mountain_areas]) + r""" \\
 & """ + ' & '.join(['\\textbf{Mean}'] * len(mountain_areas)) + r""" \\
\hline
"""
for index, row in final_summary_stats.iterrows():
    latex_table += f"{row['Variable']} & " + ' & '.join([f"{row.get(area, '')}" for area in mountain_areas]) + r" \\\\" + "\n"
latex_table += r"""\hline
\end{tabular}
\caption{Summary Statistics with Mountain Areas}
\label{tab:summary_stats}
\end{table}
"""

print(latex_table)

# Save LaTeX table to a .tex file
with open('summary_stats_table.tex', 'w') as file:
    file.write(latex_table)

print("LaTeX table saved as 'summary_stats_table.tex'")


# Join the lines back into a single string
latex_table = '\n'.join(latex_table_lines)

# Replace '.0' with '' in the 'Observations' and 'No. of Counties' rows
latex_table = latex_table.replace('147.0', '147').replace('267.0', '267').replace('282.0', '282') \
                         .replace('270.0', '270').replace('126.0', '126').replace('219.0', '219') \
                         .replace('435.0', '435').replace('115.0', '115').replace('124.0', '124') \
                         .replace('26193.0', '26193').replace('7.0', '7').replace('13.0', '13') \
                         .replace('14.0', '14').replace('6.0', '6').replace('11.0', '11') \
                         .replace('22.0', '22').replace('1284.0', '1284')

# # Add \resizebox to make the table text-wide
# latex_table = (
#     "\\begin{table}[!htbp]\n"
#     "\\centering\n"
#     "\\caption{Descriptive Statistics}\n"
#     "\\label{t:descriptive_statistics}\n"
#     "\\resizebox{\\textwidth}{!}{%\n"
#     + latex_table +
#     "}\n\\end{table}"
# )

# Print the LaTeX code
print(latex_table)

# Save the LaTeX table to a .tex file
file_path = './code/writing/tables/mountain_summary_statistics.tex'
with open(file_path, 'w') as file:
    file.write(latex_table)



#%% baseline regression

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen forest = land_use_2/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in forest {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Baseline Results: Forest Share"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab forest_spec1 forest_spec2 forest_spec3 forest_spec4 using "outputs/baseline_results2.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.4f) se(%8.4f) ///
    title("\textsc{Baseline Results: Forest Share}} \\ \label{t:baseline_results") ///
	keep(d) ///
    stats(y_mean N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Mean County Area (km2)" "Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%s %9.0fc %9.3fc %s ))
''')


#%% event study- forest share
stata.pdataframe_to_data(df, True)
stata.run('''
cap drop y
gen y = land_use_2/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"


// Estimation with did_imputation of Borusyak et al. (2021)
# did_imputation y i t ei, allhorizons pretrend(11) delta(1) autosample
# event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
#     title("Borusyak et al. (2021) imputation estimator") xlabel(-11(1)9))
# 
# estimates store bjs // storing the estimates for later
# 
# # // Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
# # did_multiplegt y i t d, robust_dynamic dynamic(8) placebo(8) breps(100) cluster(i) 
# # event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
# #     title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-8(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together
# # 
# # matrix dcdh_b = e(estimates) // storing the estimates for later
# # matrix dcdh_v = e(variances)

# // Estimation with csdid of Callaway and Sant'Anna (2020)
# cap drop gvar
# gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
# csdid y `controls', ivar(i) time(t) gvar(gvar) long2
# estat event, estore(cs) // this produces and stores the estimates at the same time
# event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
#     title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/11 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

# eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
# event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
#     title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together

# matrix sa_b = e(b_iw) // storing the estimates for later
# matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event , a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-11(1)9) ///
    title("OLS"))

estimates store ols // saving the estimates for later

event_plot ols, ///
    stub_lag(L#event Tp#) stub_lead(F#event Tm#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event Study on Forest Share", size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ylabel(-0.005(0.005)0.02) ///
     legend(order(1 "TWFE OLS Estimator") rows(1) ///
    region(style(none)) position(0.5)) ///
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(forest_green)) lag_ci_opt1(color(forest_green)) 
    
# event_plot ols cs, ///
#     stub_lag(L#event Tp#) stub_lead(F#event Tm#) plottype(scatter) ciplottype(rcap) ///
#     together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
#     graph_opt(title("Event Study on Forest Share", size(medlarge)) ///
#        xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ylabel(-0.005(0.005)0.02) ///
#        legend(order(1 "TWFE OLS" 3 "Callaway-Sant'Anna") rows(1) ///
#         region(style(none)) position(0.5)) ///
#        xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
#     ) ///
#     lag_opt1(msymbol(+) color(ltblue)) lag_ci_opt1(color(ltblue)) ///
#     lag_opt2(msymbol(Th) color(forest_green)) lag_ci_opt2(color(forest_green))
    
# // Combine all plots using the stored estimates
# event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v  bjs, ///
#     stub_lag( L#event Effect_# Tp# L#event  tau#) stub_lead(F#event Placebo_# Tm# F#event  pre#) plottype(scatter) ciplottype(rcap) ///
#     together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
#     graph_opt(title("Event Study on Forest Share", size(medlarge)) ///
#        xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ylabel(-0.005(0.005)0.025) ///
#        legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
#         region(style(none)) position(0.5)) /// This places the legend in the upper left corner
#     /// the following lines replace default_look with something more elaborate
#        xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
#     ) ///
#     lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
#     lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
#     lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
#     lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
#     lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///

graph export "outputs/event_plot_forest_only1estimator.png" , replace width(6000) height(4000)
''')

# Method 1: Display SVG directly (if you want to display SVG format)
display(SVG(filename='outputs/event_plot.svg'))

# Method 2: Display PNG or other formats using PIL (if exported as PNG)
img = Image.open('outputs/event_plot.png')
img.show()


#%% spatial heterogenetiy

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen forest = land_use_2/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.
encode mountain_area, gen(mountain_area_num)

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in forest {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' i.d##i.mountain_area_num, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' i.d##i.mountain_area_num , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' i.d##i.mountain_area_num, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' i.d##i.mountain_area_num `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Baseline Results: Forest Share"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab forest_spec1 forest_spec2 forest_spec3 forest_spec4 using "outputs/heterogeneity_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("{\textsc{Spatial Heterogeneity Effects: Forest Share}}}") ///
	keep(1.d#*mountain_area_num) ///
    stats(N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%9.0fc %9.3fc %s ))
''')



# Read a .lex file as plain text
with open('outputs/heterogeneity_results.tex', 'r', encoding='utf-8') as file:
    content = file.read()
print(content)

translation = {
    '乌蒙山区': 'Wumeng Mountain Area',
    '六盘山区': 'Liupan Mountain Area',
    '大兴安岭南麓山区': 'Southern Daxing’anling Mountain Area',
    '大别山区': 'Dabie Mountain Area',
    '武陵山区': 'Wuling Mountain Area',
    '滇桂黔石漠化区': 'Dian-Gui-Qian Karst Region',
    '滇西边境山区': 'Western Yunnan Border Mountain Area',
    '燕山-太行山区': 'Yanshan-Taihang Mountain Area',
    '秦巴山区': 'Qinba Mountain Area',
    '罗霄山区': 'Luoxiao Mountain Area'
}

# Apply translations
for chinese_name, english_name in translation.items():
    content = content.replace(chinese_name, english_name)  # Replace each occurrence

print(content)
old_text = "\caption{{    extsc{Spatial Heterogeneity Effects: Forest Share}}}}"
new_text = r"\caption{\textsc{Spatial Heterogeneity Effects: Forest Share}}"
content = content.replace(old_text, new_text)
new_line = r"\label{t:heterogeneity_effects}"

old_text = "Post-Poverty Alleviation=1 $\\times$ nan&       0.000         &       0.000         &       0.000         &       0.000         \\"
new_text = ""
content = content.replace(old_text, new_text)
old_text = "&         (.)         &         (.)         &         (.)         &         (.)         \\ [1em]"
new_text = ""
content = content.replace(old_text, new_text)
old_text = "forest"
new_text = ""
content = content.replace(old_text, new_text)

old_text = "Post-Poverty Alleviation=1"
new_text = ""
content = content.replace(old_text, new_text)

old_text = r"$\times$ "
new_text = ""
content = content.replace(old_text, new_text)

# Save the modified content back to the same file (or a new file)
with open('outputs/heterogeneity_results.tex', 'w', encoding='utf-8') as file:
    file.write(content)

with open('code/writing/tables/heterogeneity_results.tex', 'w', encoding='utf-8') as file:
    file.write(content)

print("Translation and replacement complete.")

# Print the content
print(content)


#%% mechanism 1 - forest gains

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t - ei 
replace exit_year = 0 if exit_year ==.

* Clear previously stored estimates
eststo clear

* Use local for controls
local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

* Label the land use variables
label variable share_land_use_12 "Forest cover gain (%) from cropland"
label variable share_land_use_32 "Forest cover gain (%) from shrub"
label variable share_land_use_42 "Forest cover gain (%) from grassland"
label variable share_land_use_52 "Forest cover gain (%) from water"
label variable share_land_use_62 "Forest cover gain (%) from snow"
label variable share_land_use_72 "Forest cover gain (%) from barren land"
label variable share_land_use_82 "Forest cover gain (%) from impervious surface"
label variable share_land_use_92 "Forest cover gain (%) from wetland"

* List of outcomes
local outcomes share_forest_gain share_land_use_12 share_land_use_32 share_land_use_42 share_land_use_52 share_land_use_62 share_land_use_72 share_land_use_82 share_land_use_92


foreach y of local outcomes {
    gen d_`y' = d
    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d_`y', absorb(provinceid t) vce(cluster cntyid)
    est store `y'_spec1

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d_`y', absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d_`y', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d_`y' `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
}
''')

stata.run('''
label drop _all
label variable d_share_forest_gain `""{bf:Total forest gains}" "(% of county area)""'
label variable d_share_land_use_12 "Cropland"
label variable d_share_land_use_32 "Shrub"
label variable d_share_land_use_42 "Grassland"
label variable d_share_land_use_52 "Water"
label variable d_share_land_use_62 "Snow"
label variable d_share_land_use_72 "Barren"
label variable d_share_land_use_82 "Impervious"
label variable d_share_land_use_92 "Wetland"
* Now generate the coefplot with the adjusted coefficients
coefplot  ///
    (share_forest_gain_spec1, omitted keep(d_share_forest_gain) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_forest_gain_spec2, omitted keep(d_share_forest_gain) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_forest_gain_spec3, omitted keep(d_share_forest_gain) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_forest_gain_spec4, omitted keep(d_share_forest_gain) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_12_spec1, omitted keep(d_share_land_use_12) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_land_use_12_spec2, omitted keep(d_share_land_use_12) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_land_use_12_spec3, omitted keep(d_share_land_use_12) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_land_use_12_spec4, omitted keep(d_share_land_use_12) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_32_spec1, omitted keep(d_share_land_use_32) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_32_spec2, omitted keep(d_share_land_use_32) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_32_spec3, omitted keep(d_share_land_use_32) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_32_spec4, omitted keep(d_share_land_use_32) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_42_spec1, omitted keep(d_share_land_use_42) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_42_spec2, omitted keep(d_share_land_use_42) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_42_spec3, omitted keep(d_share_land_use_42) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_42_spec4, omitted keep(d_share_land_use_42) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_52_spec1, omitted keep(d_share_land_use_52) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_52_spec2, omitted keep(d_share_land_use_52) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_52_spec3, omitted keep(d_share_land_use_52) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_52_spec4, omitted keep(d_share_land_use_52) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_62_spec1, omitted keep(d_share_land_use_62) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_62_spec2, omitted keep(d_share_land_use_62) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_62_spec3, omitted keep(d_share_land_use_62) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_62_spec4, omitted keep(d_share_land_use_62) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_72_spec1, omitted keep(d_share_land_use_72) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_72_spec2, omitted keep(d_share_land_use_72) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_72_spec3, omitted keep(d_share_land_use_72) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_72_spec4, omitted keep(d_share_land_use_72) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_82_spec1, omitted keep(d_share_land_use_82) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_82_spec2, omitted keep(d_share_land_use_82) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_82_spec3, omitted keep(d_share_land_use_82) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_82_spec4, omitted keep(d_share_land_use_82) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_92_spec1, omitted keep(d_share_land_use_92) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_92_spec2, omitted keep(d_share_land_use_92) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_92_spec3, omitted keep(d_share_land_use_92) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_92_spec4, omitted keep(d_share_land_use_92) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)), ///
    bgcolor(white) plotregion(color(white)) graphregion(color(white)) ///
    xtitle("Coefficients", size(small)) legend(rows(1) position(6) order(2 "Province FE" 4 "Prefectural City FE" 6 ///
    "County FE" 8 "County FE With Controls") size(medsmall)) ///
    xline(0, lwidth(thin)) nooffsets grid(w) ///
    coeflabels(, labsize(medium)) ///
    groups(d_share_land_use_12 d_share_land_use_22 d_share_land_use_32 d_share_land_use_42 d_share_land_use_52 ///
       d_share_land_use_62 d_share_land_use_72 d_share_land_use_82 d_share_land_use_92 = `""{bf:Forest Gains from}" "(% of county area)""', angle(90)) ///
    xscale(r(-0.001 0.001)) xtick(-0.0005(0.0005)0.002)
graph export "outputs/graph_forest_gains.jpg", as(jpg) replace width(5000) height(4000)
''')

#headings(d_share_land_use_12=" {bf:Forest Gains from}") // /
#groups(d_share_land_use_12 d_share_land_use_22 d_share_land_use_32 d_share_land_use_42 d_share_land_use_52 ///
#    d_share_land_use_62 d_share_land_use_72 d_share_land_use_82 d_share_land_use_92 = `"Forest Gains from"',
#    angle(90)) ///
#     headings(d_share_land_use_12=" {bf:from}") ///

#%% mechanism 1- forest loses

stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop i 
gen i = cntyid
cap drop t
gen t = year
cap drop d
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
cap drop ei
gen ei = 2011 if treated == 1.
cap drop k
gen k = t - ei 
replace exit_year = 0 if exit_year ==.

* Clear previously stored estimates
* eststo clear

* Use local for controls
local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

* Label the land use variables
label variable share_forest_loss `""{bf:Total forest losses}" "(% of county area)""'
label variable share_land_use_21 "Forest cover loss (%) to cropland"
label variable share_land_use_23 "Forest cover gain (%) from shrub"
label variable share_land_use_24 "Forest cover gain (%) from grassland"
label variable share_land_use_25 "Forest cover gain (%) from water"
label variable share_land_use_26 "Forest cover gain (%) from snow"
label variable share_land_use_27 "Forest cover gain (%) from barren land"
label variable share_land_use_28 "Forest cover gain (%) from impervious surface"
label variable share_land_use_29 "Forest cover gain (%) from wetland"
label variable net_share_forest_gain "New forest gain"

* List of outcomes
local outcomes share_forest_loss share_land_use_21 share_land_use_23 share_land_use_24 share_land_use_25 ///
share_land_use_26 share_land_use_27 share_land_use_28 share_land_use_29 net_share_forest_gain

foreach y of local outcomes {
    gen d_`y' = d
    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d_`y', absorb(provinceid t) vce(cluster cntyid)
    est store `y'_spec1

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d_`y', absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d_`y', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d_`y' `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
}
''')

stata.run('''
label drop _all
label variable d_share_forest_loss `""{bf:Total forest losses}" "(% of county area)""'
label variable d_share_land_use_21 "Cropland"
label variable d_share_land_use_23 "Shrub"
label variable d_share_land_use_24 "Grassland"
label variable d_share_land_use_25 "Water"
label variable d_share_land_use_26 "Snow"
label variable d_share_land_use_27 "Barren"
label variable d_share_land_use_28 "Impervious"
label variable d_share_land_use_29 "Wetland"
label variable net_share_forest_gain `""{bf:Net forest gains}" "(% of county area)""'
* Now generate the coefplot with the adjusted coefficients
coefplot  ///
    (share_forest_loss_spec1, omitted keep(d_share_forest_loss) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_forest_loss_spec2, omitted keep(d_share_forest_loss) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_forest_loss_spec3, omitted keep(d_share_forest_loss) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_forest_loss_spec4, omitted keep(d_share_forest_loss) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_21_spec1, omitted keep(d_share_land_use_21) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_land_use_21_spec2, omitted keep(d_share_land_use_21) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_land_use_21_spec3, omitted keep(d_share_land_use_21) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_land_use_21_spec4, omitted keep(d_share_land_use_21) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_23_spec1, omitted keep(d_share_land_use_23) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_23_spec2, omitted keep(d_share_land_use_23) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_23_spec3, omitted keep(d_share_land_use_23) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_23_spec4, omitted keep(d_share_land_use_23) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_24_spec1, omitted keep(d_share_land_use_24) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_24_spec2, omitted keep(d_share_land_use_24) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_24_spec3, omitted keep(d_share_land_use_24) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_24_spec4, omitted keep(d_share_land_use_24) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_25_spec1, omitted keep(d_share_land_use_25) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_25_spec2, omitted keep(d_share_land_use_25) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_25_spec3, omitted keep(d_share_land_use_25) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_25_spec4, omitted keep(d_share_land_use_25) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_26_spec1, omitted keep(d_share_land_use_26) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_26_spec2, omitted keep(d_share_land_use_26) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_26_spec3, omitted keep(d_share_land_use_26) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_26_spec4, omitted keep(d_share_land_use_26) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_27_spec1, omitted keep(d_share_land_use_27) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_27_spec2, omitted keep(d_share_land_use_27) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_27_spec3, omitted keep(d_share_land_use_27) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_27_spec4, omitted keep(d_share_land_use_27) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_28_spec1, omitted keep(d_share_land_use_28) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_28_spec2, omitted keep(d_share_land_use_28) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_28_spec3, omitted keep(d_share_land_use_28) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_28_spec4, omitted keep(d_share_land_use_28) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_29_spec1, omitted keep(d_share_land_use_29) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_29_spec2, omitted keep(d_share_land_use_29) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_29_spec3, omitted keep(d_share_land_use_29) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_29_spec4, omitted keep(d_share_land_use_29) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)), ///
        bgcolor(white) plotregion(color(white)) graphregion(color(white)) ///
    xtitle("Coefficients", size(small)) legend(rows(1) position(6) order(2 "Province FE" 4 "Prefectural City FE" 6 ///
    "County FE" 8 "County FE With Controls") size(medsmall)) ///
    xline(0, lwidth(thin)) nooffsets grid(w) ///
    coeflabels(, labsize(medium)) ///
    groups(d_share_land_use_21 d_share_land_use_23 d_share_land_use_24 d_share_land_use_25 ///
             d_share_land_use_26 d_share_land_use_27 d_share_land_use_28 d_share_land_use_29  = `""{bf:Forest Losses to}" "(% of county area)""', angle(90)) ///
    xscale(r(-0.001 0.001)) xtick(-0.0005(0.0005)0.002)
graph export "outputs/graph_forest_losses.jpg", as(jpg) replace width(5000) height(4000)
''')


#     bgcolor(white) plotregion(color(white)) graphregion(color(white)) ///
#     xtitle("Coefficients", size(small)) legend(rows(1) position(6) order(2 "Province FE" 4 "Prefectural City FE" 6 "County FE" 8 "County FE With Controls") size(small)) ///
#     xline(0, lwidth(thin)) nooffsets grid(w) ///
#     headings(d_share_land_use_21=" {bf:Forest Losses to}") ///
#     coeflabels(, labsize(vsmall)) ///
#     xscale(r(-0.001 0.0005)) xtick(-0.001(0.001)0.0005) xlabel(-.001 -.0005 0 0.0005, labsize(small))
# graph export "outputs/graph_forest_losses.jpg", as(jpg) replace width(5000) height(4000)
# ''')

#%% -mechanism 1 gains and losses

## add total forest gains and losses

stata.run('''
label drop _all
label variable d_share_forest_gain `""{bf:Total forest gains}" "(% of county area)""'
label variable d_share_land_use_12 "Cropland"
label variable d_share_land_use_32 "Shrub"
label variable d_share_land_use_42 "Grassland"
label variable d_share_land_use_52 "Water"
label variable d_share_land_use_62 "Snow"
label variable d_share_land_use_72 "Barren"
label variable d_share_land_use_82 "Impervious"
label variable d_share_land_use_92 "Wetland"
label variable d_share_forest_loss `""{bf:Total forest losses}" "(% of county area)""'
label variable d_share_land_use_21 "Cropland"
label variable d_share_land_use_23 "Shrub"
label variable d_share_land_use_24 "Grassland"
label variable d_share_land_use_25 "Water"
label variable d_share_land_use_26 "Snow"
label variable d_share_land_use_27 "Barren"
label variable d_share_land_use_28 "Impervious"
label variable d_share_land_use_29 "Wetland"
label variable d_net_share_forest_gain `""{bf:Net forest gains}" "(% of county area)""'
* Now generate the coefplot with the adjusted coefficients
graph set window fontface "Arial"
graph set window fontscale 1.2  // This adjusts to approximately 12pt font size
coefplot  ///
    (share_forest_gain_spec1, omitted keep(d_share_forest_gain) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_forest_gain_spec2, omitted keep(d_share_forest_gain) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_forest_gain_spec3, omitted keep(d_share_forest_gain) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_forest_gain_spec4, omitted keep(d_share_forest_gain) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_12_spec1, omitted keep(d_share_land_use_12) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_land_use_12_spec2, omitted keep(d_share_land_use_12) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_land_use_12_spec3, omitted keep(d_share_land_use_12) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_land_use_12_spec4, omitted keep(d_share_land_use_12) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_32_spec1, omitted keep(d_share_land_use_32) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_32_spec2, omitted keep(d_share_land_use_32) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_32_spec3, omitted keep(d_share_land_use_32) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_32_spec4, omitted keep(d_share_land_use_32) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_42_spec1, omitted keep(d_share_land_use_42) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_42_spec2, omitted keep(d_share_land_use_42) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_42_spec3, omitted keep(d_share_land_use_42) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_42_spec4, omitted keep(d_share_land_use_42) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_52_spec1, omitted keep(d_share_land_use_52) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_52_spec2, omitted keep(d_share_land_use_52) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_52_spec3, omitted keep(d_share_land_use_52) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_52_spec4, omitted keep(d_share_land_use_52) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_62_spec1, omitted keep(d_share_land_use_62) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_62_spec2, omitted keep(d_share_land_use_62) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_62_spec3, omitted keep(d_share_land_use_62) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_62_spec4, omitted keep(d_share_land_use_62) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_72_spec1, omitted keep(d_share_land_use_72) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_72_spec2, omitted keep(d_share_land_use_72) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_72_spec3, omitted keep(d_share_land_use_72) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_72_spec4, omitted keep(d_share_land_use_72) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_82_spec1, omitted keep(d_share_land_use_82) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_82_spec2, omitted keep(d_share_land_use_82) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_82_spec3, omitted keep(d_share_land_use_82) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_82_spec4, omitted keep(d_share_land_use_82) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_92_spec1, omitted keep(d_share_land_use_92) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_92_spec2, omitted keep(d_share_land_use_92) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_92_spec3, omitted keep(d_share_land_use_92) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_92_spec4, omitted keep(d_share_land_use_92) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_forest_loss_spec1, omitted keep(d_share_forest_loss) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_forest_loss_spec2, omitted keep(d_share_forest_loss) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_forest_loss_spec3, omitted keep(d_share_forest_loss) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_forest_loss_spec4, omitted keep(d_share_forest_loss) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_21_spec1, omitted keep(d_share_land_use_21) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_land_use_21_spec2, omitted keep(d_share_land_use_21) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_land_use_21_spec3, omitted keep(d_share_land_use_21) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_land_use_21_spec4, omitted keep(d_share_land_use_21) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_23_spec1, omitted keep(d_share_land_use_23) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_23_spec2, omitted keep(d_share_land_use_23) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_23_spec3, omitted keep(d_share_land_use_23) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_23_spec4, omitted keep(d_share_land_use_23) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_24_spec1, omitted keep(d_share_land_use_24) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_24_spec2, omitted keep(d_share_land_use_24) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_24_spec3, omitted keep(d_share_land_use_24) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_24_spec4, omitted keep(d_share_land_use_24) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_25_spec1, omitted keep(d_share_land_use_25) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_25_spec2, omitted keep(d_share_land_use_25) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_25_spec3, omitted keep(d_share_land_use_25) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_25_spec4, omitted keep(d_share_land_use_25) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_26_spec1, omitted keep(d_share_land_use_26) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_26_spec2, omitted keep(d_share_land_use_26) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_26_spec3, omitted keep(d_share_land_use_26) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_26_spec4, omitted keep(d_share_land_use_26) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_27_spec1, omitted keep(d_share_land_use_27) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_27_spec2, omitted keep(d_share_land_use_27) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_27_spec3, omitted keep(d_share_land_use_27) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_27_spec4, omitted keep(d_share_land_use_27) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_28_spec1, omitted keep(d_share_land_use_28) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_28_spec2, omitted keep(d_share_land_use_28) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_28_spec3, omitted keep(d_share_land_use_28) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_28_spec4, omitted keep(d_share_land_use_28) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_29_spec1, omitted keep(d_share_land_use_29) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_29_spec2, omitted keep(d_share_land_use_29) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_29_spec3, omitted keep(d_share_land_use_29) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_29_spec4, omitted keep(d_share_land_use_29) mcolor(forest_green) ciopts(lcolor(forest_green)) ///
    offset(-0.2)), ///
    bgcolor(white) plotregion(color(white)) graphregion(color(white)) ///
    xtitle("Coefficients", size(small)) legend(rows(1) position(6) order(2 "Province FE" 4 "Prefectural City FE" 6 "County FE" 8 "County FE With Controls") size(small)) ///
    xline(0, lcolor(red) lwidth(thin) lpattern(solid)) nooffsets grid(w) ///
    groups(d_share_land_use_12 d_share_land_use_22 d_share_land_use_32 d_share_land_use_42 d_share_land_use_52 ///
       d_share_land_use_62 d_share_land_use_72 d_share_land_use_82 d_share_land_use_92 = `""{bf:Forest Gains from}" "(% of county area)""' ///
       d_share_land_use_21 d_share_land_use_23 d_share_land_use_24 d_share_land_use_25 ///
       d_share_land_use_26 d_share_land_use_27 d_share_land_use_28 d_share_land_use_29 = `""{bf:Forest Losses to}" "(% of county area)""', ///
       angle(90)) ///
    coeflabels(, labsize(medium)) ///
    xscale(r(-0.0015 0.0015)) xtick(-0.0015(0.0005)0.0015) xlabel(-.0015 -.001 -.0005 0 .0005 0.001 0.0015, labsize(small))
graph export "outputs/graph_forest_gain_and_loss.jpg", as(jpg) replace width(5000) height(4000)
''')

# headings(d_share_land_use_12=" {bf:Forest Gains from}" // /
#                              d_share_land_use_21 = "{bf:Forest Losses to  }", nogap) // /

# (net_share_forest_gain_spec1, omitted keep(d_net_share_forest_gain) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
# (net_share_forest_gain_spec2, omitted keep(d_net_share_forest_gain) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
# (net_share_forest_gain_spec3, omitted keep(d_net_share_forest_gain) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
# (net_share_forest_gain_spec4, omitted keep(d_net_share_forest_gain) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)), ///
#%% mechanism 2 reforestation

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen outcome = reforestation/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in outcome {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Carbon Storage Results"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab outcome_spec1 outcome_spec2 outcome_spec3 outcome_spec4 using "outputs/forestation_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("textsc{Government Forestation Program}} \\ \label{t:forestation_program") ///
    keep(d) ///
    stats(y_mean N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Mean County Area (km2)" "Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%s %9.0fc %9.3fc %s ))
''')

#event study
stata.pdataframe_to_data(df, True)
stata.run('''
cap drop y
gen outcome = reforestation/total_area
gen y = outcome
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

// Estimation with did_imputation of Borusyak et al. (2021)
did_imputation y i t ei, allhorizons pretrend(11) delta(1) autosample
event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("Borusyak et al. (2021) imputation estimator") xlabel(-11(1)9))

estimates store bjs // storing the estimates for later

// Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
did_multiplegt y i t d, robust_dynamic dynamic(8) placebo(8) breps(100) cluster(i) 
event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-8(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together

matrix dcdh_b = e(estimates) // storing the estimates for later
matrix dcdh_v = e(variances)

// Estimation with cldid of Callaway and Sant'Anna (2020)
cap drop gvar
gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
csdid y, ivar(i) time(t) gvar(gvar)
estat event, estore(cs) // this produces and stores the estimates at the same time
event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/11 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together


matrix sa_b = e(b_iw) // storing the estimates for later
matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event, a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-11(1)9) ///
    title("OLS"))

estimates store ols // saving the estimates for later

// Combine all plots using the stored estimates
event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v  bjs, ///
    stub_lag( L#event Effect_# Tp# L#event  tau#) stub_lead(F#event Placebo_# Tm# F#event  pre#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event Study Estimators on Planted Forest Share from Gov't Forestation Initiatives", ///
                    size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ///
ylabel(-0.01(0.005)0.01) ///
       legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
        region(style(none)) position(0.5)) /// This places the legend in the upper left corner
    /// the following lines replace default_look with something more elaborate
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
    lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
    lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
    lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
    lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///

graph export "outputs/event_plot_forestation.png" , replace width(6000) height(4000)
''')

# Method 1: Display SVG directly (if you want to display SVG format)
display(SVG(filename='outputs/event_plot_reforestation.png'))

# Method 2: Display PNG or other formats using PIL (if exported as PNG)
img = Image.open('outputs/event_plot_reforestation.png')
img.show()






#%% mechnism 3 rural village

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen forest = ur_code_220
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in forest {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$checkmark$", replace
    estadd local year_fe "$checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d, absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Forest Outcome Variables"
label variable d "Participation in the Poverty Alleviation Program"

* Use formatting for Observations with thousand separator
esttab forest_spec1 forest_spec2 forest_spec3 forest_spec4 using "outputs/table_rural_number.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("Regression Results") ///
   keep(d) ///
    order(d) ///
    collabels(none) ///
    modelwidth(15) ///
    stats(N r2 province_fe city_fe cnty_fe year_fe has_controls y_mean, ///
          l("Observations" "R-squared" "Province FE" "Prefectural City FE" "County FE" "Year FE" "Controls" "Average County Area (km2)") ///
          fmt(%9.0fc %9.3fc %s %s))
''')





#%% event study on rural number change
stata.pdataframe_to_data(df, True)
stata.run('''
cap drop i 
gen i = cntyid
cap drop t
gen t = year
cap drop d
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
cap drop ei
gen ei = 2011 if treated == 1.
cap drop k
gen k = t - ei 
replace exit_year = 0 if exit_year ==.

cap drop y 
gen y = log(ur_code_220)

# // Estimation with did_imputation of Borusyak et al. (2021)
# did_imputation y i t ei, allhorizons pretrend(2) delta(1) autosample
# event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
#     title("Borusyak et al. (2021) imputation estimator") xlabel(-2(1)9))
# 
# estimates store bjs // storing the estimates for later

# // Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
# did_multiplegt y i t d, robust_dynamic dynamic(9) placebo(1) breps(100) cluster(i) 
# event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
#     title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-2(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together
# 
# matrix dcdh_b = e(estimates) // storing the estimates for later
# matrix dcdh_v = e(variances)

# // Estimation with csdid of Callaway and Sant'Anna (2020)
# cap drop gvar
# gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
# csdid y, ivar(i) time(t) gvar(gvar) long2
# estat event, estore(cs) // this produces and stores the estimates at the same time
# event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-2(1)9) ///
#     title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort 
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/2 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

# eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
# event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-2(1)9) ///
#     title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together
# 
# 
# matrix sa_b = e(b_iw) // storing the estimates for later
# matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event, a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-2(1)9) ///
    title("OLS"))
estimates store ols // saving the estimates for later

event_plot ols, ///
    stub_lag(L#event Tp#) stub_lead(F#event Tm#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event Study on Rural Village Number Change", size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-2(1)9) ylabel(-0.1(0.05)0.1) ///
       legend(order(1 "TWFE OLS") rows(1) ///
        region(style(none)) position(0.5)) ///
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(forest_green)) lag_ci_opt1(color(forest_green))
// Export the graph
graph export "outputs/event_plot_villages_only1estimator.png", replace width(6000) height(4000)

# // Combine all plots using the stored estimates
# event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v bjs, ///
#     stub_lag(L#event Effect_# Tp# L#event tau#) ///
#     stub_lead(F#event Placebo_# Tm# F#event pre#) ///
#     plottype(scatter) ciplottype(rcap) together ///
#     perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
#     graph_opt(title("Event Study on Changes in Rural Village Numbers", size(medlarge)) ///
#         xtitle("Years since the event (2011)") ///
#         ytitle("Average effect") ///
#         xscale(range(-2 9)) xlabel(-2(1)9) ///
#         ylabel(-0.1(0.05)0.1) ///
#         legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
#             region(style(none)) position(0.5)) ///
#         xline(-0.5, lcolor(gs8) lpattern(dash)) ///
#         yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ///
#         ylabel(, angle(horizontal)) ///
#     ) ///
#     lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
#     lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
#     lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
#     lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
#     lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///
# // Export the graph
# graph export "outputs/event_plot_villages.png", replace width(6000) height(4000)
''')

stata.pdataframe_to_data(df, True)

stata.run('''
local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"
reg share_land_use_2 ur_code_220 i.year i.cntyid
''')

# Filter data for 'treated' == 1 and 'year' >= 2011
data = df[(df['treated'] == 1) & (df['year'] >= 2011)]

# Calculate yearly averages
yearly_avg = data.groupby('year')[['ur_code_220', 'share_land_use_2']].mean().reset_index()

# Calculate correlation between 'ur_code_220' and 'share_land_use_2'
correlation = yearly_avg['ur_code_220'].corr(yearly_avg['share_land_use_2'])

# Create figure and plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot UR Code 220 (left y-axis)
ax1.plot(yearly_avg['year'], yearly_avg['ur_code_220'], color='black', marker='o', linestyle='-', label="Average UR Code 220")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Number of Rural Villages", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(150, 220)  # Set limits for left y-axis

# Second y-axis for Share of Land Use Type 2 (right y-axis)
ax2 = ax1.twinx()
ax2.plot(yearly_avg['year'], yearly_avg['share_land_use_2'], color='black', marker='s', linestyle='-', label="Average Share of Land Use Type 2")
ax2.set_ylabel("Average Forest Share", color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0.47, 0.52)  # Set limits for right y-axis

# Adding title, grid, and correlation
plt.title("Average County Forest Share and Number of Rural Villages\nand their Correlation in Treatment Group (2011-2020)", fontsize=14)
plt.grid(True)

# Add the correlation text in black at the top left
plt.text(0.15, 0.82, f"Correlation: {correlation:.2f}", transform=fig.transFigure, fontsize=14, color="black")

# Add text above each line to label them directly
ax1.text(yearly_avg['year'].iloc[-5], yearly_avg['ur_code_220'].iloc[-1] - 6, "Average Number of Rural Villages",
         color='black', fontsize=14)
ax2.text(yearly_avg['year'].iloc[-4], yearly_avg['share_land_use_2'].iloc[-7] , "Average Forest Share",
         color='black', fontsize=14)
plt.savefig("/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest/code/writing/figures/correlation.png", dpi=300, bbox_inches='tight')
plt.show()


# Filter data for 'treated' == 1 and 'year' >= 2011
data = df[(df['treated'] == 1) & (df['year'] >= 2011)]

# Create scatter plot for ur_code_220 vs. share_land_use_2
plt.figure(figsize=(8, 6))
plt.scatter(data['ur_code_220'], data['share_land_use_2'], color='black', alpha=0.7)

# Calculate correlation between 'ur_code_220' and 'share_land_use_2'
correlation = data['ur_code_220'].corr(data['share_land_use_2'])

# Add labels, title, and correlation text
plt.xlabel("Number of Rural Villages")
plt.ylabel("Forest Share")
plt.title("Number of Rural Villages and Forest Share in Treatment Group (2011-2020)")
plt.text(0.05, 0.9, f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, color="black", bbox=dict(facecolor='white', edgecolor='black'))

plt.savefig("code/writing/figures/scatter_plot.png", dpi=300, bbox_inches='tight')

plt.grid(True)
plt.legend()
plt.show()

#%%% channel impervious surface

# Filter data for 'treated' == 1 and 'year' >= 2011
data = df[(df['treated'] == 1) & (df['year'] >= 2011)]


# Calculate yearly averages
yearly_avg = data.groupby('year')[['ur_code_220', 'share_land_use_8']].mean().reset_index()


# Calculate correlation between 'ur_code_220' and 'share_land_use_8'
correlation = yearly_avg['ur_code_220'].corr(yearly_avg['share_land_use_8'])


# Create figure and plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))


# Plot UR Code 220 (left y-axis)
ax1.plot(yearly_avg['year'], yearly_avg['ur_code_220'], color='black', marker='o', linestyle='-', label="Average UR Code 220")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Number of Rural Villages", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(150, 220)  # Set limits for left y-axis


# Second y-axis for Share of Land Use Type 2 (right y-axis)
ax2 = ax1.twinx()
ax2.plot(yearly_avg['year'], yearly_avg['share_land_use_8'], color='black', marker='s', linestyle='-', label="Average Share of Land Use Type 2")
ax2.set_ylabel("Average Forest Share", color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0.47, 0.52)  # Set limits for right y-axis


# Adding title, grid, and correlation
plt.title("Average County Forest Share and Number of Rural Villages\nand their Correlation in Treatment Group (2011-2020)", fontsize=14)
plt.grid(True)


# Add the correlation text in black at the top left
plt.text(0.15, 0.82, f"Correlation: {correlation:.2f}", transform=fig.transFigure, fontsize=14, color="black")


# Add text above each line to label them directly
ax1.text(yearly_avg['year'].iloc[-5], yearly_avg['ur_code_220'].iloc[-1] - 6, "Average Number of Rural Villages",
        color='black', fontsize=14)
ax2.text(yearly_avg['year'].iloc[-4], yearly_avg['share_land_use_8'].iloc[-7] , "Average Forest Share",
        color='black', fontsize=14)
#plt.savefig("/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest/code/writing/figures"
             #"/correlation.png", dpi=300, bbox_inches='tight')
plt.show()




# Filter data for 'treated' == 1 and 'year' >= 2011
data = df[(df['treated'] == 1) & (df['year'] >= 2011)]

# Calculate yearly averages
yearly_avg = data.groupby('year')[['share_land_use_8', 'share_land_use_2']].mean().reset_index()

# Calculate correlation between 'share_land_use_8' and 'share_land_use_2'
correlation = yearly_avg['share_land_use_8'].corr(yearly_avg['share_land_use_2'])

# Create figure and plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Share of Land Use Type 8 (left y-axis)
ax1.plot(yearly_avg['year'], yearly_avg['share_land_use_8'], color='black', marker='o', linestyle='-', label="Average Share of Land Use Type 8")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Share of Land Use Type 8", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(yearly_avg['share_land_use_8'].min() * 0.9, yearly_avg['share_land_use_8'].max() * 1.1)

# Second y-axis for Share of Land Use Type 2 (right y-axis)
ax2 = ax1.twinx()
ax2.plot(yearly_avg['year'], yearly_avg['share_land_use_2'], color='black', marker='s', linestyle='-', label="Average Share of Land Use Type 2")
ax2.set_ylabel("Average Forest Share (Land Use Type 2)", color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(yearly_avg['share_land_use_2'].min() * 0.9, yearly_avg['share_land_use_2'].max() * 1.1)

# Adding title, grid, and correlation
plt.title("Average Forest Share and Land Use Type 8 Share\nand their Correlation in Treatment Group (2011-2020)", fontsize=14)
plt.grid(True)

# Add the correlation text in black at the top left
plt.text(0.15, 0.82, f"Correlation: {correlation:.2f}", transform=fig.transFigure, fontsize=14, color="black")

# Add text above each line to label them directly
ax1.text(yearly_avg['year'].iloc[-1], yearly_avg['share_land_use_8'].iloc[-1] + 0.02, "Average Share of Land Use Type 8", color='black', fontsize=12)
ax2.text(yearly_avg['year'].iloc[-1], yearly_avg['share_land_use_2'].iloc[-1] - 0.01, "Average Forest Share", color='black', fontsize=12)

# Save and show the plot
plt.savefig("/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest/code/writing/figures/correlation_impervious.png", dpi=300, bbox_inches='tight')
plt.show()


#%% carbon

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen outcome = carbon_value/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in outcome {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Carbon Storage Results"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab outcome_spec1 outcome_spec2 outcome_spec3 outcome_spec4 using "outputs/carbon_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("\textsc{Carbon Storage Results}} \\ \label{t:carbon_results") ///
    keep(d) ///
    stats(y_mean N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Mean County Area (km2)" "Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%s %9.0fc %9.3fc %s ))
''')

#event study
stata.pdataframe_to_data(df, True)
stata.run('''
cap drop y
gen outcome = carbon_value/total_area
gen y = outcome
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

// Estimation with did_imputation of Borusyak et al. (2021)
did_imputation y i t ei, allhorizons pretrend(11) delta(1) autosample
event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("Borusyak et al. (2021) imputation estimator") xlabel(-11(1)9))

estimates store bjs // storing the estimates for later

// Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
did_multiplegt y i t d, robust_dynamic dynamic(8) placebo(8) breps(100) cluster(i) 
event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-8(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together

matrix dcdh_b = e(estimates) // storing the estimates for later
matrix dcdh_v = e(variances)

// Estimation with cldid of Callaway and Sant'Anna (2020)
cap drop gvar
gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
csdid y, ivar(i) time(t) gvar(gvar)
estat event, estore(cs) // this produces and stores the estimates at the same time
event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/11 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together


matrix sa_b = e(b_iw) // storing the estimates for later
matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event, a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-11(1)9) ///
    title("OLS"))

estimates store ols // saving the estimates for later


// Combine all plots using the stored estimates
event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v  bjs, ///
    stub_lag( L#event Effect_# Tp# L#event  tau#) stub_lead(F#event Placebo_# Tm# F#event  pre#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event study estimators on carbon storage", size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect (tons C per square kilometer)") xlabel(-11(1)9) ylabel(-200(200)600) ///
       legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
        region(style(none)) position(0.5)) /// This places the legend in the upper left corner
    /// the following lines replace default_look with something more elaborate
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
    lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
    lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
    lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
    lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///

graph export "outputs/event_plot_carbon.png" , replace width(6000) height(4000)
''')

# Method 1: Display SVG directly (if you want to display SVG format)
display(SVG(filename='outputs/event_plot_carbon.png'))

# Method 2: Display PNG or other formats using PIL (if exported as PNG)
img = Image.open('outputs/event_plot_carbon.png')
img.show()


#%% all land use change
stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"

* Drop variables if they exist and generate new ones
cap drop i 
gen i = cntyid
cap drop t
gen t = year
cap drop d
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.

cap drop ei
gen ei = 2011 if treated == 1

cap drop k
gen k = t - ei 

replace exit_year = 0 if exit_year ==.

* Clear previously stored estimates
* eststo clear

* Use local for controls
local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

local outcomes ///
    share_land_use_12 share_land_use_13 share_land_use_14 share_land_use_15 share_land_use_16 share_land_use_17 share_land_use_18 share_land_use_19 ///
    share_land_use_21 share_land_use_23 share_land_use_24 share_land_use_25 share_land_use_26 share_land_use_27 share_land_use_28 share_land_use_29 ///
    share_land_use_31 share_land_use_32 share_land_use_34 share_land_use_35 share_land_use_36 share_land_use_37 share_land_use_38 share_land_use_39 ///
    share_land_use_41 share_land_use_42 share_land_use_43 share_land_use_45 share_land_use_46 share_land_use_47 share_land_use_48 share_land_use_49 ///
    share_land_use_51 share_land_use_52 share_land_use_53 share_land_use_54 share_land_use_56 share_land_use_57 share_land_use_58 share_land_use_59 ///
    share_land_use_61 share_land_use_62 share_land_use_63 share_land_use_64 share_land_use_65 share_land_use_67 share_land_use_68 share_land_use_69 ///
    share_land_use_71 share_land_use_72 share_land_use_73 share_land_use_74 share_land_use_75 share_land_use_76 share_land_use_78 share_land_use_79 ///
    share_land_use_81 share_land_use_82 share_land_use_83 share_land_use_84 share_land_use_85 share_land_use_86 share_land_use_87 share_land_use_89 ///
    share_land_use_91 share_land_use_92 share_land_use_93 share_land_use_94 share_land_use_95 share_land_use_96 share_land_use_97 share_land_use_98

foreach y of local outcomes {
    cap drop d_`y'
    gen d_`y' = d
    // Run basic regression without controls
    reghdfe `y' d_`y', absorb(provinceid t) vce(cluster cntyid)
    est store `y'_spec1

    reghdfe `y' d_`y', absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2

    reghdfe `y' d_`y', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3

    // Run full model with controls
    reghdfe `y' d_`y' `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
}

* Label the land use variables
label variable d_share_land_use_12 "Cropland to Forest"
label variable d_share_land_use_13 "Cropland to Shrub"
label variable d_share_land_use_14 "Cropland to Grassland"
label variable d_share_land_use_15 "Cropland to Water"
label variable d_share_land_use_16 "Cropland to Snow"
label variable d_share_land_use_17 "Cropland to Barren land"
label variable d_share_land_use_18 "Cropland to Impervious surface"
label variable d_share_land_use_19 "Cropland to Wetland"

label variable d_share_land_use_21 "Forest to Cropland"
label variable d_share_land_use_23 "Forest to Shrub"
label variable d_share_land_use_24 "Forest to Grassland"
label variable d_share_land_use_25 "Forest to Water"
label variable d_share_land_use_26 "Forest to Snow"
label variable d_share_land_use_27 "Forest to Barren land"
label variable d_share_land_use_28 "Forest to Impervious surface"
label variable d_share_land_use_29 "Forest to Wetland"

label variable d_share_land_use_31 "Shrub to Cropland"
label variable d_share_land_use_32 "Shrub to Forest"
label variable d_share_land_use_34 "Shrub to Grassland"
label variable d_share_land_use_35 "Shrub to Water"
label variable d_share_land_use_36 "Shrub to Snow"
label variable d_share_land_use_37 "Shrub to Barren land"
label variable d_share_land_use_38 "Shrub to Impervious surface"
label variable d_share_land_use_39 "Shrub to Wetland"

label variable d_share_land_use_41 "Grassland to Cropland"
label variable d_share_land_use_42 "Grassland to Forest"
label variable d_share_land_use_43 "Grassland to Shrub"
label variable d_share_land_use_45 "Grassland to Water"
label variable d_share_land_use_46 "Grassland to Snow"
label variable d_share_land_use_47 "Grassland to Barren land"
label variable d_share_land_use_48 "Grassland to Impervious surface"
label variable d_share_land_use_49 "Grassland to Wetland"

label variable d_share_land_use_51 "Water to Cropland"
label variable d_share_land_use_52 "Water to Forest"
label variable d_share_land_use_53 "Water to Shrub"
label variable d_share_land_use_54 "Water to Grassland"
label variable d_share_land_use_56 "Water to Snow"
label variable d_share_land_use_57 "Water to Barren land"
label variable d_share_land_use_58 "Water to Impervious surface"
label variable d_share_land_use_59 "Water to Wetland"

label variable d_share_land_use_61 "Snow to Cropland"
label variable d_share_land_use_62 "Snow to Forest"
label variable d_share_land_use_63 "Snow to Shrub"
label variable d_share_land_use_64 "Snow to Grassland"
label variable d_share_land_use_65 "Snow to Water"
label variable d_share_land_use_67 "Snow to Barren land"
label variable d_share_land_use_68 "Snow to Impervious surface"
label variable d_share_land_use_69 "Snow to Wetland"

label variable d_share_land_use_71 "Barren land to Cropland"
label variable d_share_land_use_72 "Barren land to Forest"
label variable d_share_land_use_73 "Barren land to Shrub"
label variable d_share_land_use_74 "Barren land to Grassland"
label variable d_share_land_use_75 "Barren land to Water"
label variable d_share_land_use_76 "Barren land to Snow"
label variable d_share_land_use_78 "Barren land to Impervious surface"
label variable d_share_land_use_79 "Barren land to Wetland"

label variable d_share_land_use_81 "Impervious surface to Cropland"
label variable d_share_land_use_82 "Impervious surface to Forest"
label variable d_share_land_use_83 "Impervious surface to Shrub"
label variable d_share_land_use_84 "Impervious surface to Grassland"
label variable d_share_land_use_85 "Impervious surface to Water"
label variable d_share_land_use_86 "Impervious surface to Snow"
label variable d_share_land_use_87 "Impervious surface to Barren land"
label variable d_share_land_use_89 "Impervious surface to Wetland"

label variable d_share_land_use_91 "Wetland to Cropland"
label variable d_share_land_use_92 "Wetland to Forest"
label variable d_share_land_use_93 "Wetland to Shrub"
label variable d_share_land_use_94 "Wetland to Grassland"
label variable d_share_land_use_95 "Wetland to Water"
label variable d_share_land_use_96 "Wetland to Snow"
label variable d_share_land_use_97 "Wetland to Barren land"
label variable d_share_land_use_98 "Wetland to Impervious surface"

coefplot  ///
    // Cropland to X (12 - 19)
    (share_land_use_12_spec1, omitted keep(d_share_land_use_12) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Forest
    (share_land_use_12_spec2, omitted keep(d_share_land_use_12) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_12_spec3, omitted keep(d_share_land_use_12) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_12_spec4, omitted keep(d_share_land_use_12) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_13_spec1, omitted keep(d_share_land_use_13) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Shrub
    (share_land_use_13_spec2, omitted keep(d_share_land_use_13) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_13_spec3, omitted keep(d_share_land_use_13) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_13_spec4, omitted keep(d_share_land_use_13) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_14_spec1, omitted keep(d_share_land_use_14) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Grassland
    (share_land_use_14_spec2, omitted keep(d_share_land_use_14) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_14_spec3, omitted keep(d_share_land_use_14) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_14_spec4, omitted keep(d_share_land_use_14) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_15_spec1, omitted keep(d_share_land_use_15) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Water
    (share_land_use_15_spec2, omitted keep(d_share_land_use_15) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_15_spec3, omitted keep(d_share_land_use_15) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_15_spec4, omitted keep(d_share_land_use_15) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_16_spec1, omitted keep(d_share_land_use_16) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Snow
    (share_land_use_16_spec2, omitted keep(d_share_land_use_16) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_16_spec3, omitted keep(d_share_land_use_16) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_16_spec4, omitted keep(d_share_land_use_16) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_17_spec1, omitted keep(d_share_land_use_17) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Barren land
    (share_land_use_17_spec2, omitted keep(d_share_land_use_17) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_17_spec3, omitted keep(d_share_land_use_17) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_17_spec4, omitted keep(d_share_land_use_17) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_18_spec1, omitted keep(d_share_land_use_18) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Impervious surface
    (share_land_use_18_spec2, omitted keep(d_share_land_use_18) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_18_spec3, omitted keep(d_share_land_use_18) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_18_spec4, omitted keep(d_share_land_use_18) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_19_spec1, omitted keep(d_share_land_use_19) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Cropland to Wetland
    (share_land_use_19_spec2, omitted keep(d_share_land_use_19) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_19_spec3, omitted keep(d_share_land_use_19) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_19_spec4, omitted keep(d_share_land_use_19) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Forest to X (21 - 29)
    (share_land_use_21_spec1, omitted keep(d_share_land_use_21) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Cropland
    (share_land_use_21_spec2, omitted keep(d_share_land_use_21) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_21_spec3, omitted keep(d_share_land_use_21) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_21_spec4, omitted keep(d_share_land_use_21) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_23_spec1, omitted keep(d_share_land_use_23) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Shrub
    (share_land_use_23_spec2, omitted keep(d_share_land_use_23) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_23_spec3, omitted keep(d_share_land_use_23) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_23_spec4, omitted keep(d_share_land_use_23) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_24_spec1, omitted keep(d_share_land_use_24) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Grassland
    (share_land_use_24_spec2, omitted keep(d_share_land_use_24) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_24_spec3, omitted keep(d_share_land_use_24) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_24_spec4, omitted keep(d_share_land_use_24) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_25_spec1, omitted keep(d_share_land_use_25) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Water
    (share_land_use_25_spec2, omitted keep(d_share_land_use_25) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_25_spec3, omitted keep(d_share_land_use_25) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_25_spec4, omitted keep(d_share_land_use_25) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_26_spec1, omitted keep(d_share_land_use_26) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Snow
    (share_land_use_26_spec2, omitted keep(d_share_land_use_26) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_26_spec3, omitted keep(d_share_land_use_26) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_26_spec4, omitted keep(d_share_land_use_26) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_27_spec1, omitted keep(d_share_land_use_27) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Barren land
    (share_land_use_27_spec2, omitted keep(d_share_land_use_27) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_27_spec3, omitted keep(d_share_land_use_27) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_27_spec4, omitted keep(d_share_land_use_27) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_28_spec1, omitted keep(d_share_land_use_28) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Impervious surface
    (share_land_use_28_spec2, omitted keep(d_share_land_use_28) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_28_spec3, omitted keep(d_share_land_use_28) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_28_spec4, omitted keep(d_share_land_use_28) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_29_spec1, omitted keep(d_share_land_use_29) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Forest to Wetland
    (share_land_use_29_spec2, omitted keep(d_share_land_use_29) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_29_spec3, omitted keep(d_share_land_use_29) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_29_spec4, omitted keep(d_share_land_use_29) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Shrub to X (31 - 39)
    (share_land_use_31_spec1, omitted keep(d_share_land_use_31) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Cropland
    (share_land_use_31_spec2, omitted keep(d_share_land_use_31) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_31_spec3, omitted keep(d_share_land_use_31) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_31_spec4, omitted keep(d_share_land_use_31) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_32_spec1, omitted keep(d_share_land_use_32) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Forest
    (share_land_use_32_spec2, omitted keep(d_share_land_use_32) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_32_spec3, omitted keep(d_share_land_use_32) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_32_spec4, omitted keep(d_share_land_use_32) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_34_spec1, omitted keep(d_share_land_use_34) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Grassland
    (share_land_use_34_spec2, omitted keep(d_share_land_use_34) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_34_spec3, omitted keep(d_share_land_use_34) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_34_spec4, omitted keep(d_share_land_use_34) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_35_spec1, omitted keep(d_share_land_use_35) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Water
    (share_land_use_35_spec2, omitted keep(d_share_land_use_35) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_35_spec3, omitted keep(d_share_land_use_35) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_35_spec4, omitted keep(d_share_land_use_35) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_36_spec1, omitted keep(d_share_land_use_36) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Snow
    (share_land_use_36_spec2, omitted keep(d_share_land_use_36) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_36_spec3, omitted keep(d_share_land_use_36) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_36_spec4, omitted keep(d_share_land_use_36) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_37_spec1, omitted keep(d_share_land_use_37) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Barren land
    (share_land_use_37_spec2, omitted keep(d_share_land_use_37) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_37_spec3, omitted keep(d_share_land_use_37) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_37_spec4, omitted keep(d_share_land_use_37) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_38_spec1, omitted keep(d_share_land_use_38) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Impervious surface
    (share_land_use_38_spec2, omitted keep(d_share_land_use_38) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_38_spec3, omitted keep(d_share_land_use_38) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_38_spec4, omitted keep(d_share_land_use_38) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_39_spec1, omitted keep(d_share_land_use_39) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Shrub to Wetland
    (share_land_use_39_spec2, omitted keep(d_share_land_use_39) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_39_spec3, omitted keep(d_share_land_use_39) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_39_spec4, omitted keep(d_share_land_use_39) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Grassland to X (41 - 49)
    (share_land_use_41_spec1, omitted keep(d_share_land_use_41) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Cropland
    (share_land_use_41_spec2, omitted keep(d_share_land_use_41) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_41_spec3, omitted keep(d_share_land_use_41) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_41_spec4, omitted keep(d_share_land_use_41) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_42_spec1, omitted keep(d_share_land_use_42) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Forest
    (share_land_use_42_spec2, omitted keep(d_share_land_use_42) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_42_spec3, omitted keep(d_share_land_use_42) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_42_spec4, omitted keep(d_share_land_use_42) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_43_spec1, omitted keep(d_share_land_use_43) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Shrub
    (share_land_use_43_spec2, omitted keep(d_share_land_use_43) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_43_spec3, omitted keep(d_share_land_use_43) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_43_spec4, omitted keep(d_share_land_use_43) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_45_spec1, omitted keep(d_share_land_use_45) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Water
    (share_land_use_45_spec2, omitted keep(d_share_land_use_45) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_45_spec3, omitted keep(d_share_land_use_45) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_45_spec4, omitted keep(d_share_land_use_45) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_46_spec1, omitted keep(d_share_land_use_46) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Snow
    (share_land_use_46_spec2, omitted keep(d_share_land_use_46) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_46_spec3, omitted keep(d_share_land_use_46) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_46_spec4, omitted keep(d_share_land_use_46) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_47_spec1, omitted keep(d_share_land_use_47) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Barren land
    (share_land_use_47_spec2, omitted keep(d_share_land_use_47) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_47_spec3, omitted keep(d_share_land_use_47) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_47_spec4, omitted keep(d_share_land_use_47) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_48_spec1, omitted keep(d_share_land_use_48) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Impervious surface
    (share_land_use_48_spec2, omitted keep(d_share_land_use_48) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_48_spec3, omitted keep(d_share_land_use_48) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_48_spec4, omitted keep(d_share_land_use_48) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_49_spec1, omitted keep(d_share_land_use_49) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Grassland to Wetland
    (share_land_use_49_spec2, omitted keep(d_share_land_use_49) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_49_spec3, omitted keep(d_share_land_use_49) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_49_spec4, omitted keep(d_share_land_use_49) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
        (share_land_use_51_spec1, omitted keep(d_share_land_use_51) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Cropland
    (share_land_use_51_spec2, omitted keep(d_share_land_use_51) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_51_spec3, omitted keep(d_share_land_use_51) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_51_spec4, omitted keep(d_share_land_use_51) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_52_spec1, omitted keep(d_share_land_use_52) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Forest
    (share_land_use_52_spec2, omitted keep(d_share_land_use_52) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_52_spec3, omitted keep(d_share_land_use_52) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_52_spec4, omitted keep(d_share_land_use_52) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_53_spec1, omitted keep(d_share_land_use_53) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Shrub
    (share_land_use_53_spec2, omitted keep(d_share_land_use_53) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_53_spec3, omitted keep(d_share_land_use_53) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_53_spec4, omitted keep(d_share_land_use_53) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_54_spec1, omitted keep(d_share_land_use_54) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Grassland
    (share_land_use_54_spec2, omitted keep(d_share_land_use_54) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_54_spec3, omitted keep(d_share_land_use_54) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_54_spec4, omitted keep(d_share_land_use_54) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_56_spec1, omitted keep(d_share_land_use_56) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Snow
    (share_land_use_56_spec2, omitted keep(d_share_land_use_56) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_56_spec3, omitted keep(d_share_land_use_56) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_56_spec4, omitted keep(d_share_land_use_56) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_57_spec1, omitted keep(d_share_land_use_57) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Barren land
    (share_land_use_57_spec2, omitted keep(d_share_land_use_57) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_57_spec3, omitted keep(d_share_land_use_57) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_57_spec4, omitted keep(d_share_land_use_57) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_58_spec1, omitted keep(d_share_land_use_58) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Impervious surface
    (share_land_use_58_spec2, omitted keep(d_share_land_use_58) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_58_spec3, omitted keep(d_share_land_use_58) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_58_spec4, omitted keep(d_share_land_use_58) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_59_spec1, omitted keep(d_share_land_use_59) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Water to Wetland
    (share_land_use_59_spec2, omitted keep(d_share_land_use_59) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_59_spec3, omitted keep(d_share_land_use_59) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_59_spec4, omitted keep(d_share_land_use_59) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Snow to X (61 - 69)
    (share_land_use_61_spec1, omitted keep(d_share_land_use_61) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Cropland
    (share_land_use_61_spec2, omitted keep(d_share_land_use_61) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_61_spec3, omitted keep(d_share_land_use_61) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_61_spec4, omitted keep(d_share_land_use_61) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_62_spec1, omitted keep(d_share_land_use_62) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Forest
    (share_land_use_62_spec2, omitted keep(d_share_land_use_62) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_62_spec3, omitted keep(d_share_land_use_62) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_62_spec4, omitted keep(d_share_land_use_62) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_63_spec1, omitted keep(d_share_land_use_63) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Shrub
    (share_land_use_63_spec2, omitted keep(d_share_land_use_63) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_63_spec3, omitted keep(d_share_land_use_63) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_63_spec4, omitted keep(d_share_land_use_63) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_64_spec1, omitted keep(d_share_land_use_64) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Grassland
    (share_land_use_64_spec2, omitted keep(d_share_land_use_64) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_64_spec3, omitted keep(d_share_land_use_64) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_64_spec4, omitted keep(d_share_land_use_64) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_65_spec1, omitted keep(d_share_land_use_65) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Water
    (share_land_use_65_spec2, omitted keep(d_share_land_use_65) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_65_spec3, omitted keep(d_share_land_use_65) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_65_spec4, omitted keep(d_share_land_use_65) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_67_spec1, omitted keep(d_share_land_use_67) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Barren land
    (share_land_use_67_spec2, omitted keep(d_share_land_use_67) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_67_spec3, omitted keep(d_share_land_use_67) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_67_spec4, omitted keep(d_share_land_use_67) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_68_spec1, omitted keep(d_share_land_use_68) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Impervious surface
    (share_land_use_68_spec2, omitted keep(d_share_land_use_68) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_68_spec3, omitted keep(d_share_land_use_68) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_68_spec4, omitted keep(d_share_land_use_68) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_69_spec1, omitted keep(d_share_land_use_69) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Snow to Wetland
    (share_land_use_69_spec2, omitted keep(d_share_land_use_69) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_69_spec3, omitted keep(d_share_land_use_69) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_69_spec4, omitted keep(d_share_land_use_69) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Barren land to X (71 - 79)
    (share_land_use_71_spec1, omitted keep(d_share_land_use_71) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Cropland
    (share_land_use_71_spec2, omitted keep(d_share_land_use_71) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_71_spec3, omitted keep(d_share_land_use_71) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_71_spec4, omitted keep(d_share_land_use_71) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_72_spec1, omitted keep(d_share_land_use_72) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Forest
    (share_land_use_72_spec2, omitted keep(d_share_land_use_72) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_72_spec3, omitted keep(d_share_land_use_72) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_72_spec4, omitted keep(d_share_land_use_72) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_73_spec1, omitted keep(d_share_land_use_73) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Shrub
    (share_land_use_73_spec2, omitted keep(d_share_land_use_73) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_73_spec3, omitted keep(d_share_land_use_73) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_73_spec4, omitted keep(d_share_land_use_73) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_74_spec1, omitted keep(d_share_land_use_74) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Grassland
    (share_land_use_74_spec2, omitted keep(d_share_land_use_74) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_74_spec3, omitted keep(d_share_land_use_74) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_74_spec4, omitted keep(d_share_land_use_74) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_75_spec1, omitted keep(d_share_land_use_75) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Water
    (share_land_use_75_spec2, omitted keep(d_share_land_use_75) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_75_spec3, omitted keep(d_share_land_use_75) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_75_spec4, omitted keep(d_share_land_use_75) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_76_spec1, omitted keep(d_share_land_use_76) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Snow
    (share_land_use_76_spec2, omitted keep(d_share_land_use_76) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_76_spec3, omitted keep(d_share_land_use_76) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_76_spec4, omitted keep(d_share_land_use_76) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_78_spec1, omitted keep(d_share_land_use_78) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Impervious surface
    (share_land_use_78_spec2, omitted keep(d_share_land_use_78) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_78_spec3, omitted keep(d_share_land_use_78) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_78_spec4, omitted keep(d_share_land_use_78) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_79_spec1, omitted keep(d_share_land_use_79) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Barren land to Wetland
    (share_land_use_79_spec2, omitted keep(d_share_land_use_79) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_79_spec3, omitted keep(d_share_land_use_79) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_79_spec4, omitted keep(d_share_land_use_79) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Impervious surface to X (81 - 89)
    (share_land_use_81_spec1, omitted keep(d_share_land_use_81) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Cropland
    (share_land_use_81_spec2, omitted keep(d_share_land_use_81) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_81_spec3, omitted keep(d_share_land_use_81) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_81_spec4, omitted keep(d_share_land_use_81) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_82_spec1, omitted keep(d_share_land_use_82) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Forest
    (share_land_use_82_spec2, omitted keep(d_share_land_use_82) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_82_spec3, omitted keep(d_share_land_use_82) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_82_spec4, omitted keep(d_share_land_use_82) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_83_spec1, omitted keep(d_share_land_use_83) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Shrub
    (share_land_use_83_spec2, omitted keep(d_share_land_use_83) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_83_spec3, omitted keep(d_share_land_use_83) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_83_spec4, omitted keep(d_share_land_use_83) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_84_spec1, omitted keep(d_share_land_use_84) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Grassland
    (share_land_use_84_spec2, omitted keep(d_share_land_use_84) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_84_spec3, omitted keep(d_share_land_use_84) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_84_spec4, omitted keep(d_share_land_use_84) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_85_spec1, omitted keep(d_share_land_use_85) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Water
    (share_land_use_85_spec2, omitted keep(d_share_land_use_85) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_85_spec3, omitted keep(d_share_land_use_85) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_85_spec4, omitted keep(d_share_land_use_85) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_86_spec1, omitted keep(d_share_land_use_86) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Snow
    (share_land_use_86_spec2, omitted keep(d_share_land_use_86) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_86_spec3, omitted keep(d_share_land_use_86) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_86_spec4, omitted keep(d_share_land_use_86) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_87_spec1, omitted keep(d_share_land_use_87) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Barren land
    (share_land_use_87_spec2, omitted keep(d_share_land_use_87) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_87_spec3, omitted keep(d_share_land_use_87) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_87_spec4, omitted keep(d_share_land_use_87) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_89_spec1, omitted keep(d_share_land_use_89) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Impervious surface to Wetland
    (share_land_use_89_spec2, omitted keep(d_share_land_use_89) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_89_spec3, omitted keep(d_share_land_use_89) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_89_spec4, omitted keep(d_share_land_use_89) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Wetland to X (91 - 98)
    (share_land_use_91_spec1, omitted keep(d_share_land_use_91) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Cropland
    (share_land_use_91_spec2, omitted keep(d_share_land_use_91) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_91_spec3, omitted keep(d_share_land_use_91) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_91_spec4, omitted keep(d_share_land_use_91) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_92_spec1, omitted keep(d_share_land_use_92) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Forest
    (share_land_use_92_spec2, omitted keep(d_share_land_use_92) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_92_spec3, omitted keep(d_share_land_use_92) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_92_spec4, omitted keep(d_share_land_use_92) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_93_spec1, omitted keep(d_share_land_use_93) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Shrub
    (share_land_use_93_spec2, omitted keep(d_share_land_use_93) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_93_spec3, omitted keep(d_share_land_use_93) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_93_spec4, omitted keep(d_share_land_use_93) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_94_spec1, omitted keep(d_share_land_use_94) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Grassland
    (share_land_use_94_spec2, omitted keep(d_share_land_use_94) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_94_spec3, omitted keep(d_share_land_use_94) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_94_spec4, omitted keep(d_share_land_use_94) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_95_spec1, omitted keep(d_share_land_use_95) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Water
    (share_land_use_95_spec2, omitted keep(d_share_land_use_95) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_95_spec3, omitted keep(d_share_land_use_95) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_95_spec4, omitted keep(d_share_land_use_95) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_96_spec1, omitted keep(d_share_land_use_96) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Snow
    (share_land_use_96_spec2, omitted keep(d_share_land_use_96) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_96_spec3, omitted keep(d_share_land_use_96) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_96_spec4, omitted keep(d_share_land_use_96) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_97_spec1, omitted keep(d_share_land_use_97) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Barren land
    (share_land_use_97_spec2, omitted keep(d_share_land_use_97) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_97_spec3, omitted keep(d_share_land_use_97) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_97_spec4, omitted keep(d_share_land_use_97) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    (share_land_use_98_spec1, omitted keep(d_share_land_use_98) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Wetland to Impervious surface
    (share_land_use_98_spec2, omitted keep(d_share_land_use_98) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) 
    (share_land_use_98_spec3, omitted keep(d_share_land_use_98) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) 
    (share_land_use_98_spec4, omitted keep(d_share_land_use_98) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) 
    ///
    // Export the graph
    bgcolor(white) plotregion(color(white)) graphregion(color(white)) ///
    xtitle("Coefficients", size(small)) ///
    legend(rows(1) position(6) order(2 "Province FE" 4 "Prefectural City FE" 6 "County FE" 8 "County FE With Controls") size(small)) ///
    xline(0, lwidth(thin)) nooffsets grid(w) ///
    coeflabels(, labsize(vsmall)) ///
    xscale(r(-0.001 0.0005)) xtick(-0.001(0.001)0.0005) xlabel(-.001 -.0005 0 0.0005, labsize(small))

graph export "outputs/graph_4.jpg", as(jpg) replace width(5000) height(4000)
''')
#%% robustness 1 check- forest share change


stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen outcome = net_share_forest_gain
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in outcome {
    * Calculate county_area (mean of total_area)
    egen net_share_forest_mean = mean(net_share_forest_gain)
    gen net_share_forest = net_share_forest_mean
    egen county_area_mean = mean(total_area)
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = net_share_forest, replace
    estadd local y_mean_county = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = net_share_forest, replace
    estadd local y_mean_county = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = net_share_forest, replace
    estadd local y_mean_county = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = net_share_forest, replace
    estadd local y_mean_county = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "forest share change"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab outcome_spec1 outcome_spec2 outcome_spec3 outcome_spec4 using "outputs/net_forest_gain_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("textsc{Carbon Storage Results}} \\ \label{t:carbon_results") ///
    keep(d) ///
    stats(N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Observations" "R-squared" "Province FE" ///
          "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%9.5f %s %9.0fc %9.3fc %s ))
''')

#     stats(y_mean y_mean_county  N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
#           l("Mean Forest Share Change" "Mean County Area (km2)" "Observations" "R-squared" "Province FE" ///
#           "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
#           fmt(%9.5f %s %9.0fc %9.3fc %s ))
# ''')

#event study
stata.pdataframe_to_data(df, True)
stata.run('''
cap drop y
gen outcome = net_share_forest_gain
gen y = outcome
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

// Estimation with did_imputation of Borusyak et al. (2021)
did_imputation y i t ei, allhorizons pretrend(11) delta(1) autosample
event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("Borusyak et al. (2021) imputation estimator") xlabel(-11(1)9))

estimates store bjs // storing the estimates for later

// Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
did_multiplegt y i t d, robust_dynamic dynamic(8) placebo(8) breps(100) cluster(i) 
event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-8(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together

matrix dcdh_b = e(estimates) // storing the estimates for later
matrix dcdh_v = e(variances)

// Estimation with cldid of Callaway and Sant'Anna (2020)
cap drop gvar
gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
csdid y, ivar(i) time(t) gvar(gvar)
estat event, estore(cs) // this produces and stores the estimates at the same time
event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/11 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together


matrix sa_b = e(b_iw) // storing the estimates for later
matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event, a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-11(1)9) ///
    title("OLS"))

estimates store ols // saving the estimates for later

// Combine all plots using the stored estimates
event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v  bjs, ///
    stub_lag( L#event Effect_# Tp# L#event  tau#) stub_lead(F#event Placebo_# Tm# F#event  pre#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event study estimators on forest share change", size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ylabel(-0.004(0.001)0.004) ///
       legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
        region(style(none)) position(0.5)) /// This places the legend in the upper left corner
    /// the following lines replace default_look with something more elaborate
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
    lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
    lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
    lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
    lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///

graph export "outputs/event_plot_net_forest_gain.png" , replace width(6000) height(4000)
''')



#%% robustness 2 check-other land use

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t - ei 
replace exit_year = 0 if exit_year ==.

* Clear previously stored estimates
eststo clear

* Use local for controls
local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

* Label the land use variables
label variable share_land_use_1 " Cropland"
label variable share_land_use_3 " shrub"
label variable share_land_use_4 "grassland"
label variable share_land_use_5 "water"
label variable share_land_use_6 "snow"
label variable share_land_use_7 "barren land"
label variable share_land_use_8 "impervious surface"
label variable share_land_use_9 "wetland"

* List of outcomes
local outcomes  share_land_use_1 share_land_use_3 share_land_use_4 share_land_use_5 share_land_use_6 share_land_use_7 share_land_use_8 share_land_use_9


foreach y of local outcomes {
    gen d_`y' = d
    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d_`y', absorb(provinceid t) vce(cluster cntyid)
    est store `y'_spec1

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d_`y', absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d_`y', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d_`y' `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
}
''')

stata.run('''
label drop _all
label variable d_share_land_use_1 "Cropland"
label variable d_share_land_use_3 "Shrub"
label variable d_share_land_use_4 "Grassland"
label variable d_share_land_use_5 "Water"
label variable d_share_land_use_6 "Snow"
label variable d_share_land_use_7 "Barren"
label variable d_share_land_use_8 "Impervious"
label variable d_share_land_use_9 "Wetland"
* Now generate the coefplot with the adjusted coefficients
coefplot  ///
    (share_land_use_1_spec1, omitted keep(d_share_land_use_1) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) /// Light Blue
    (share_land_use_1_spec2, omitted keep(d_share_land_use_1) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) /// Mid Blue
    (share_land_use_1_spec3, omitted keep(d_share_land_use_1) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) /// Mid Green
    (share_land_use_1_spec4, omitted keep(d_share_land_use_1) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) /// Forest Green
    (share_land_use_3_spec1, omitted keep(d_share_land_use_3) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_3_spec2, omitted keep(d_share_land_use_3) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_3_spec3, omitted keep(d_share_land_use_3) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_3_spec4, omitted keep(d_share_land_use_3) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_4_spec1, omitted keep(d_share_land_use_4) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_4_spec2, omitted keep(d_share_land_use_4) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_4_spec3, omitted keep(d_share_land_use_4) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_4_spec4, omitted keep(d_share_land_use_4) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_5_spec1, omitted keep(d_share_land_use_5) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_5_spec2, omitted keep(d_share_land_use_5) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_5_spec3, omitted keep(d_share_land_use_5) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_5_spec4, omitted keep(d_share_land_use_5) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_6_spec1, omitted keep(d_share_land_use_6) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_6_spec2, omitted keep(d_share_land_use_6) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_6_spec3, omitted keep(d_share_land_use_6) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_6_spec4, omitted keep(d_share_land_use_6) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_7_spec1, omitted keep(d_share_land_use_7) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_7_spec2, omitted keep(d_share_land_use_7) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_7_spec3, omitted keep(d_share_land_use_7) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_7_spec4, omitted keep(d_share_land_use_7) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_8_spec1, omitted keep(d_share_land_use_8) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_8_spec2, omitted keep(d_share_land_use_8) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_8_spec3, omitted keep(d_share_land_use_8) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_8_spec4, omitted keep(d_share_land_use_8) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)) ///
    (share_land_use_9_spec1, omitted keep(d_share_land_use_9) mcolor(ltblue) ciopts(lcolor(ltblue)) offset(0.2)) ///
    (share_land_use_9_spec2, omitted keep(d_share_land_use_9) mcolor(midblue) ciopts(lcolor(midblue)) offset(0.1)) ///
    (share_land_use_9_spec3, omitted keep(d_share_land_use_9) mcolor(midgreen) ciopts(lcolor(midgreen)) offset(-0.1)) ///
    (share_land_use_9_spec4, omitted keep(d_share_land_use_9) mcolor(forest_green) ciopts(lcolor(forest_green)) offset(-0.2)), ///
    bgcolor(white) plotregion(color(white)) graphregion(color(white)) ///
    xtitle("Coefficients", size(small)) legend(rows(1) position(6) order(2 "Province FE" 4 "Prefectural City FE" 6 ///
    "County FE" 8 "County FE With Controls") size(medsmall)) ///
    xline(0, lwidth(thin)) nooffsets grid(w) ///
    coeflabels(, labsize(medium)) ///
    groups(d_share_land_use_1 d_share_land_use_2 d_share_land_use_3 d_share_land_use_4 d_share_land_use_5 ///
       d_share_land_use_6 d_share_land_use_7 d_share_land_use_8 d_share_land_use_9 = `""{bf:Land Share of }" "(% of county area)""', angle(90)) ///
    xscale(r(-0.15 0.05)) xtick(-0.15(0.05)0.05)
graph export "outputs/graph_other_land_use.jpg", as(jpg) replace width(5000) height(4000)
''')

#     headings(d_share_land_use_1=" {bf:Share of}") ///



#%% robustness 3 check-NDVI

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen outcome = NDVI_mean/10000
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in outcome{
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Baseline Results: Forest Share"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab outcome_spec1 outcome_spec2 outcome_spec3 outcome_spec4 using "outputs/ndvi_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("\textsc{Baseline Results: Forest Share}} \\ \label{t:baseline_results") ///
    keep(d) ///
    stats(y_mean N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Mean County Area (km2)" "Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%s %9.0fc %9.3fc %s ))
''')



stata.pdataframe_to_data(df, True)
stata.run('''
cap drop y
gen y = NDVI_mean/10000
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

// Estimation with did_imputation of Borusyak et al. (2021)
did_imputation y i t ei, allhorizons pretrend(11) delta(1) autosample
event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("Borusyak et al. (2021) imputation estimator") xlabel(-11(1)9))

estimates store bjs // storing the estimates for later

// Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
did_multiplegt y i t d, robust_dynamic dynamic(8) placebo(8) breps(100) cluster(i) 
event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-8(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together

matrix dcdh_b = e(estimates) // storing the estimates for later
matrix dcdh_v = e(variances)

// Estimation with cldid of Callaway and Sant'Anna (2020)
cap drop gvar
gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
csdid y, ivar(i) time(t) gvar(gvar)
estat event, estore(cs) // this produces and stores the estimates at the same time
event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/11 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together


matrix sa_b = e(b_iw) // storing the estimates for later
matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event, a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-11(1)9) ///
    title("OLS"))

estimates store ols // saving the estimates for later

// Combine all plots using the stored estimates
event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v  bjs, ///
    stub_lag( L#event Effect_# Tp# L#event  tau#) stub_lead(F#event Placebo_# Tm# F#event  pre#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event study estimators on NDVI", size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ylabel(-0.05(0.01)0.05) ///
       legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
        region(style(none)) position(0.5)) /// This places the legend in the upper left corner
    /// the following lines replace default_look with something more elaborate
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
    lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
    lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
    lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
    lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///

graph export "outputs/event_plot_ndvi.png" , replace width(6000) height(4000)
''')



#%% imperious surface

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen forest = land_use_8/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall rainfall_square wind_speed wind_speed_square"

foreach y in forest {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Baseline Results: Forest Share"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab forest_spec1 forest_spec2 forest_spec3 forest_spec4 using "outputs/impervious_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.4f) se(%8.4f) ///
    title("\textsc{Baseline Results: Forest Share}} \\ \label{t:baseline_results") ///
	keep(d) ///
    stats(y_mean N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Mean County Area (km2)" "Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%s %9.0fc %9.3fc %s ))
''')


#%% event study
stata.pdataframe_to_data(df, True)
stata.run('''
cap drop y
gen y = land_use_8/total_area
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

// Estimation with did_imputation of Borusyak et al. (2021)
did_imputation y i t ei, allhorizons pretrend(11) delta(1) autosample
event_plot, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("Borusyak et al. (2021) imputation estimator") xlabel(-11(1)9))

estimates store bjs // storing the estimates for later

// Estimation with did_multiplegt of de Chaisemartin and d'Haultfoeuille (2020)
did_multiplegt y i t d, robust_dynamic dynamic(8) placebo(8) breps(100) cluster(i) 
event_plot e(estimates)#e(variances), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") ///
    title("de Chaisemartin and d'Haultfoeuille (2020)") xlabel(-8(1)9)) stub_lag(Effect_#) stub_lead(Placebo_#) together

matrix dcdh_b = e(estimates) // storing the estimates for later
matrix dcdh_v = e(variances)

// Estimation with cldid of Callaway and Sant'Anna (2020)
cap drop gvar
gen gvar = cond(ei==., 0, ei) // group variable as required for the csdid command
csdid y, ivar(i) time(t) gvar(gvar)
estat event, estore(cs) // this produces and stores the estimates at the same time
event_plot cs, default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Callaway and Sant'Anna (2020)")) stub_lag(Tp#) stub_lead(Tm#) together

// Estimation with eventstudyinteract of Sun and Abraham (2020)
sum ei
sum k
cap drop lastcohort
gen lastcohort = ei==r(max) // dummy for the latest- or never-treated cohort
gen never_treat = ei ==.
cap drop L*
forvalues l = 0/9 {
    gen L`l'event = k ==`l'
}
cap drop F*
forvalues l = 1/11 {
    gen F`l'event = k ==-`l'
}
drop F1event // normalize k=-1 (and also k=-15) to zero

eventstudyinteract y L*event F*event, vce(cluster i) absorb(i t) cohort(ei) control_cohort(never_treat)
event_plot e(b_iw)#e(V_iw), default_look graph_opt(xtitle("Periods since the event") ytitle("Average causal effect") xlabel(-11(1)9) ///
    title("Sun and Abraham (2020)")) stub_lag(L#event) stub_lead(F#event) together


matrix sa_b = e(b_iw) // storing the estimates for later
matrix sa_v = e(V_iw)

// TWFE OLS estimation (which is correct here because of treatment effect homogeneity). Some groups could be binned.
reghdfe y F*event L*event, a(i t) cluster(i)
event_plot, default_look stub_lag(L#event) stub_lead(F#event) together graph_opt(xtitle("Periods since the event") ytitle("OLS coefficients") xlabel(-11(1)9) ///
    title("OLS"))

estimates store ols // saving the estimates for later

// Combine all plots using the stored estimates
event_plot ols dcdh_b#dcdh_v cs sa_b#sa_v  bjs, ///
    stub_lag( L#event Effect_# Tp# L#event  tau#) stub_lead(F#event Placebo_# Tm# F#event  pre#) plottype(scatter) ciplottype(rcap) ///
    together perturb(-0.325(0.13)0.325) trimlead(11) noautolegend ///
    graph_opt(title("Event study estimators on impervious surface share", size(medlarge)) ///
       xtitle("Years since the event (2011)") ytitle("Average effect") xlabel(-11(1)9) ylabel(-0.02(0.01)0.02) ///
       legend(order(1 "TWFE OLS" 3 "de Chaisemartin-d'Haultfoeuille" 5 "Callaway-Sant'Anna" 7 "Sun-Abraham" 9 "Borusyak et al.") rows(2) ///
        region(style(none)) position(0.5)) /// This places the legend in the upper left corner
    /// the following lines replace default_look with something more elaborate
       xline(-0.5, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
    ) ///
    lag_opt1(msymbol(+) color(purple)) lag_ci_opt1(color(purple)) ///
    lag_opt2(msymbol(Oh) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
    lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
    lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
    lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange)) ///

graph export "outputs/event_plot_imperious.png" , replace width(6000) height(4000)
''')

#%%

df['urban_ratio'] = pd.to_numeric(df['urban_ratio'], errors='coerce')

stata.pdataframe_to_data(df, True)
stata.run('''
cd "/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest"
cap drop y
gen outcome = urban_ratio
gen i = cntyid
gen t = year
gen d = 1 if treated == 1 & year >= 2011
replace d = 0 if d ==.
gen ei = 2011 if treated == 1.
gen k = t-ei 
replace exit_year = 0 if exit_year ==.

local controls "population vad_pri vad_sec rdls soc_wels gov_rev gov_exp savings lightmean rainfall 
rainfall_square wind_speed wind_speed_square"

foreach y in outcome {
    * Calculate county_area (mean of total_area)
    egen county_area_mean = mean(total_area)

    * Then, round the result using the round() function
    gen county_area = round(county_area_mean)

    * Spec 1 - Basic regression with province and year fixed effects
    reghdfe `y' d, absorb(provinceid t) vce(cluster cntyid)
    display "Spec 1 finished for `y'"
    est store `y'_spec1
    estadd local y_mean = county_area, replace
    estadd local province_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 2 - Regression with city and year fixed effects
    reghdfe `y' d , absorb(cityid t) vce(cluster cntyid)
    est store `y'_spec2
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local cnty_fe "", replace
    estadd local has_controls "", replace

    * Spec 3 - Regression with county and year fixed effects
    reghdfe `y' d, absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec3
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "", replace

    * Spec 4 - Full model with additional control variables and county/year fixed effects
    reghdfe `y' d `controls', absorb(cntyid t) vce(cluster cntyid)
    est store `y'_spec4
    estadd local y_mean = county_area, replace
    estadd local province_fe "", replace
    estadd local city_fe "", replace
    estadd local cnty_fe "$\checkmark$", replace
    estadd local year_fe "$\checkmark$", replace
    estadd local has_controls "$\checkmark$", replace
}

* Generate the summary table using esttab
local title = "Carbon Storage Results"
label variable d "Post-Poverty Alleviation"

* Use formatting for Observations with thousand separator
esttab outcome_spec1 outcome_spec2 outcome_spec3 outcome_spec4 using "outputs/urban_ratio_results.tex", replace ///
    label se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%8.3f) se(%8.3f) ///
    title("textsc{Government Forestation Program}} \\ \label{t:forestation_program") ///
    keep(d) ///
    stats(y_mean N r2 province_fe city_fe cnty_fe year_fe has_controls , ///
          l("Mean County Area (km2)" "Observations" "R-squared" "Province FE" "Prefectural-City FE" "County FE" "Year FE" "Controls" ) ///
          fmt(%s %9.0fc %9.3fc %s ))
''')


#%% fund program trends

import matplotlib.pyplot as plt

# Data
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
total_fund = [4.05, 5.09, 6.06, 6.65, 7.10, 10.22, 13.18, 16.25, 19.32, 22.40]
industrial = [1.84, 2.30, 2.71, 2.99, 3.20, 4.60, 5.93, 7.31, 8.55, 9.97]
poverty_relocation = [0.77, 0.97, 1.15, 1.26, 1.35, 1.99, 2.64, 3.25, 3.94, 4.48]
education = [0.54, 0.64, 0.77, 0.86, 0.92, 1.07, 1.46, 1.84, 2.07, 2.38]
healthcare = [0.31, 0.43, 0.49, 0.54, 0.58, 0.77, 1.07, 1.30, 1.53, 1.84]
housing_renovation = [0.38, 0.46, 0.54, 0.60, 0.64, 0.92, 1.18, 1.53, 1.84, 2.12]
infrastructure = [0.15, 0.23, 0.31, 0.34, 0.32, 0.69, 0.72, 0.86, 1.10, 1.24]
social_security = [0.06, 0.06, 0.09, 0.08, 0.08, 0.18, 0.18, 0.15, 0.28, 0.38]

# RGB color for dark blue (3, 65, 151) converted to [0, 1] scale
dark_blue = (3/255, 65/255, 151/255)

# Plotting
plt.figure(figsize=(12, 8))
# #plt.plot(years, total_fund, label='Total Fund', color=dark_blue, linestyle='-', marker='o')
# plt.plot(years, industrial, label='Industrial Development', color=dark_blue, linestyle='-', marker='s')
# plt.plot(years, poverty_relocation, label='Poverty Relocation', color=dark_blue, linestyle='-', marker='^')
# plt.plot(years, education, label='Education Support', color=dark_blue, linestyle='-', marker='v')
# plt.plot(years, healthcare, label='Healthcare Support', color=dark_blue, linestyle='-', marker='D')
# plt.plot(years, housing_renovation, label='Housing & Renovation', color=dark_blue, linestyle='-', marker='*')
# plt.plot(years, infrastructure, label='Infrastructure Development', color=dark_blue, linestyle='-', marker='p')
# plt.plot(years, social_security, label='Social Security', color=dark_blue, linestyle='-', marker='x')
plt.plot(years, industrial, label='Agricultural Support ', color='black', linestyle='-', marker='s')
plt.plot(years, poverty_relocation, label='Poverty Alleviation through Relocation', color='black', linestyle='-',
         marker='^')
plt.plot(years, education, label='Education Support', color='black', linestyle='-', marker='v')
plt.plot(years, healthcare, label='Healthcare Support', color='black', linestyle='-', marker='D')
plt.plot(years, housing_renovation, label='Housing & Renovation', color='black', linestyle='-', marker='*')
plt.plot(years, infrastructure, label='Infrastructure Development', color='black', linestyle='-', marker='p')
plt.plot(years, social_security, label='Social Security', color='black', linestyle='-', marker='x')


# Customizing the graph
plt.title("Central Government’s Special Fund for Poverty Alleviation Spending (2011-2020)", fontsize=16,
          fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Billion USD', fontsize=14)
plt.grid(True, linestyle='-', alpha=0.5, axis='y')  # Horizontal grid lines only
plt.legend(loc='upper left', fontsize=12, frameon=False)  # Remove box from legend
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Display the graph
plt.tight_layout()
plt.savefig('/Users/long/Library/CloudStorage/OneDrive-Personal/Projects/poverty_forest/code/writing/figures'
            '/poverty_alleviation_fund_spending.png', dpi=600)
plt.show()

#%% land use change portion

cols_21_to_29 = [f'share_land_use_{i}' for i in range(21, 30) if i != 22]
 # Columns from 21 to 29
cols_ending_2 = [f'share_land_use_{i}' for i in range(12, 100, 10) if i != 22]  # Columns 12, 32, ..., 92

# Combine the lists of columns
selected_cols = cols_21_to_29 + cols_ending_2

# Filter only the columns that exist in the DataFrame
selected_cols = [col for col in selected_cols if col in df.columns]

cols_i_j = [f'share_land_use_{i}{j}' for i in range(1, 10) for j in range(1, 10) if i != j]


sum_selected_cols = df.groupby('year')[selected_cols].sum().sum(axis=1)
sum_cols_i_j = df.groupby('year')[cols_i_j].sum().sum(axis=1)

# Calculate the proportion of the sum of selected_cols over the sum of cols_i_j
proportion = sum_selected_cols / sum_cols_i_j

# Create a DataFrame to display the results
result_df = pd.DataFrame({
    'sum_selected_cols': sum_selected_cols,
    'sum_cols_i_j': sum_cols_i_j,
    'proportion': proportion
})




