"""test script for addl data"""

import matplotlib.pyplot as plt
import pandas as pd
import pickle

# path_and_filename = "./data/API_CM.MKT.LCAP.GD.ZS_DS2_en_excel_v2_604 - cleaned.xls"
# country_data = pd.read_excel(path_and_filename, sheet_name='Data', skiprows=4).iloc[:, [1,2]]

# country_dict_ISO_3_to_2 = {}
# for row in country_data.itertuples():
#     country_dict_ISO_3_to_2[row[2]] = row[1]

country_dict_df = pd.read_csv('./data/country_dict_ISO_3_to_2.csv')
country_dict_df.columns = ['ISO_3', 'ISO_2']
country_dict_ISO_3_to_2 = dict(zip(
    country_dict_df['ISO_3'], country_dict_df['ISO_2']
))

df = pd.read_csv('./data/OECD.SDD.STES,DSD_STES@DF_CLI,+all.csv')

df = df[
    (df['FREQ'] == "M") & 
    (df['MEASURE'] == 'LI') & 
    (df['ADJUSTMENT'] == 'AA')
].sort_values(by=[
    'REF_AREA', 
    'TIME_PERIOD'
], ascending=True)

cols_to_select = [
    'REF_AREA',
    'TIME_PERIOD',
    'OBS_VALUE',
]
df = df[cols_to_select]
df['REF_AREA'] = df['REF_AREA'].map(country_dict_ISO_3_to_2)
df = df.dropna(subset=['REF_AREA']) # 13 of original 16 countries remaining

df = df.rename(columns={'TIME_PERIOD': 'Date'}) 

df['Date'] = pd.to_datetime(df['Date']) + pd.offsets.MonthEnd(0) # convert bom to eom

df = pd.pivot(df, columns='REF_AREA', index='Date', values='OBS_VALUE').reset_index()

# plt.figure(figsize=(12, 6))
# for ref_area in df['REF_AREA'].unique():
#     subset = df[df['REF_AREA'] == ref_area]
#     plt.plot(subset['TIME_PERIOD'], subset['OBS_VALUE'], label=ref_area, linewidth=0.5)

# plt.xlabel('Time Period')
# plt.ylabel('Observation Value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
