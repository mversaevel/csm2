""" Python script for event study analysis """

import ast
import matplotlib.pyplot as plt
import pandas as pd

data_path = 'output/run4_1-mimicking'
data_path_RC1 = 'output/run4_1_RC1-mimicking'
data_path_RC2 = 'output/run4-mimicking'

trad_country_returns = pd.read_csv(data_path + '/trad_country_returns.csv')
trad_country_returns['Date'] = pd.to_datetime(trad_country_returns['Date'])
trad_country_returns = trad_country_returns.set_index('Date')

trad_country_returns_EW = pd.read_csv(data_path + '/trad_country_returns_EW.csv')
trad_country_returns_EW['Date'] = pd.to_datetime(trad_country_returns_EW['Date'])
trad_country_returns_EW = trad_country_returns_EW.set_index('Date')

csm_returns = pd.read_csv(data_path + '/csm_returns.csv')
csm_returns['Date'] = pd.to_datetime(csm_returns['Date'])
csm_returns = csm_returns.set_index('Date')

data_gpr = pd.read_excel('data' + '/data_gpr_export.xls')

country_dict_df = pd.read_csv('data/country_dict_ISO_3_to_2.csv')
country_dict_df.columns = ['ISO_3', 'ISO_2']
country_dict_ISO_3_to_2 = dict(zip(
    country_dict_df['ISO_3'], country_dict_df['ISO_2']
))

optim_description = pd.read_csv(data_path + '/optim_description.csv').iloc[:, 1:]
COUNTRIES_OF_INTEREST = optim_description.loc[optim_description['Unnamed: 0'] == 'countries_for_optimization', '0'].values[0]
COUNTRIES_OF_INTEREST = ast.literal_eval(COUNTRIES_OF_INTEREST)

### Prepare geopolitical risk index data

data_GPRC = data_gpr[
    data_gpr.columns[data_gpr.columns.str.contains('GPRC_|month')]
].copy().dropna()

data_GPRC.columns = data_GPRC.columns.str.split('_').str[-1] # drops GPRC_ prefix and retains only countries code

data_GPRC['Date'] = pd.to_datetime(data_GPRC['month']) + pd.offsets.MonthEnd(0) # pd.offsets.Day(0) # transform from start of month to previous month end
data_GPRC = data_GPRC.drop(columns='month')
data_GPRC = data_GPRC.set_index('Date')

# drop columns for which countries are missing:
matching_countries = data_GPRC.columns[[False if pd.isna(col) else True for col in data_GPRC.columns.map(country_dict_ISO_3_to_2)]]
data_GPRC = data_GPRC[matching_countries]

# then convert 3-letter ISO country code to 2-letter ISO:
data_GPRC = data_GPRC.rename(country_dict_ISO_3_to_2, axis=1)

# Then filter on relevant date_range
data_GPRC = data_GPRC.loc[csm_returns.index]

# Find, per country, the top-n events
n_events = 3
t_event_window = 12
event_dictionary: dict[list] = {}
# slice t_event_window months from the df on both sides, so that event study has sufficient data
data_GPRC_for_events = data_GPRC.iloc[t_event_window:(data_GPRC.shape[0]-t_event_window), :].copy()

for country in COUNTRIES_OF_INTEREST:
    top_events = data_GPRC_for_events.nlargest(n_events, country).index.to_list()
    event_dictionary[country] = top_events

event_returns = {}
event_returns_relative = {}
for country, event_list in event_dictionary.items():
    event_returns_relative_by_country = []
    for event in event_list:
        start_date = event - pd.offsets.MonthEnd(t_event_window)
        end_date = event + pd.offsets.MonthEnd(t_event_window)

        trad_return_series = trad_country_returns[country].loc[start_date:end_date]
        trad_return_series_cumulative = (1 + trad_return_series).cumprod() - 1
        trad_return_series_cumulative_centered = trad_return_series_cumulative - trad_return_series_cumulative.loc[event]
        trad_return_series_cumulative_centered.index = [f't={i}' for i in range(-t_event_window, t_event_window+1)]

        csm_return_series = csm_returns[country].loc[start_date:end_date]
        csm_return_series_cumulative = (1 + csm_return_series).cumprod() - 1
        csm_return_series_cumulative_centered = csm_return_series_cumulative - csm_return_series_cumulative.loc[event]
        csm_return_series_cumulative_centered.index = [f't={i}' for i in range(-t_event_window, t_event_window+1)]

        event_relative_returns = (1 + csm_return_series) / (1 + trad_return_series) - 1
        event_relative_returns_cumulative = (1 + event_relative_returns).cumprod() - 1
        event_relative_returns_cumulative_centered = event_relative_returns_cumulative - event_relative_returns_cumulative.loc[event]
        event_relative_returns_cumulative_centered.index = [f't={i}' for i in range(-t_event_window, t_event_window+1)]

        combined_return_series = pd.concat([
            trad_return_series_cumulative_centered,
            csm_return_series_cumulative_centered,
        ], axis=1, keys=['trad_returns', 'csm_returns'])

        event_returns[f'{country}-event-{event}'] = combined_return_series
        event_returns_relative_by_country.append(event_relative_returns_cumulative_centered)
    
    event_returns_relative[country] = pd.concat(event_returns_relative_by_country, axis=1).mean(axis=1)
      
# ### grid plots for n = 1 (largest event per country)

# n_plots = len(event_returns_relative)

# # Create a grid of subplots (adjust nrows/ncols as needed)
# fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 12))
# axes = axes.flatten()  # Flatten to easily iterate

# for idx, (name, df) in enumerate(event_returns_relative.items()):
#     df = pd.DataFrame(df)
#     ax = axes[idx]
#     ax.plot(df.index, df, label=df.columns[0])
#     # ax.plot(df.index, df.iloc[:, 0], label=df.columns[0])
#     # ax.plot(df.index, df.iloc[:, 1], label=df.columns[1])
#     ax.set_title(name)
#     ax.legend()
#     ax.grid(True)

#    # Add horizontal line at y=0
#     ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    
#     # Find the position of 't=0' and add vertical line there
#     if 't=0' in df.index:
#         t0_position = list(df.index).index('t=0')
#         ax.axvline(x=t0_position, color='black', linewidth=1, linestyle='-')
    
#     # Remove top and right spines
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.show()


### single plot for all effects combined; allows multiple events (n > 1)
combined_country_events = pd.DataFrame(pd.concat(event_returns_relative, axis=1).mean(axis=1))

# Create a grid of subplots (adjust nrows/ncols as needed)
fig, ax = plt.subplots(figsize=(12,12))

ax.plot(combined_country_events.index, combined_country_events, label=combined_country_events.columns[0])
# ax.set_title()
ax.grid(True)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=1, linestyle='-')

# Find the position of 't=0' and add vertical line there
if 't=0' in combined_country_events.index:
    t0_position = list(combined_country_events.index).index('t=0')
    ax.axvline(x=t0_position, color='black', linewidth=1, linestyle='-')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()










