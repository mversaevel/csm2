""" collection of functions """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco

from scipy.stats import f
from typing import Literal

def create_efficient_frontier(
        mean_returns: np.ndarray, 
        cov_matrix: pd.DataFrame, 
        target_ret_normalizer: int = 1000,
        n_points: int = 150,
) -> pd.DataFrame:
    """ Create input for plotting an efficient frontier 
    
    Note: heavily inspired by https://www.quantifiedstrategies.com/mean-variance-portfolio-in-python/
    """

    # helper functions for optimizer:
    def _port_returns(weights):
        mean_returns = args[0]
        port_returns = mean_returns.dot(weights)
        return port_returns

    def _port_vol(weights, *args):
        covariance_returns = args[1]
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_returns, weights))) # np.dot performs much faster than @ for many small matrix multiplications
        return port_vol

    # starting weights:
    starting_weights = np.array([1.0 / len(mean_returns)] * len(mean_returns))

    # bounds:
    lower_bound = 0.0
    upper_bound = 1.0
    bounds = tuple((lower_bound, upper_bound) for _ in range(len(mean_returns)))

    # args: 
    args = (mean_returns, cov_matrix)

    # collect and store results: 
    frontier_returns = []
    frontier_vols = []

    for loop_return in range(n_points):
        target_return = loop_return / target_ret_normalizer
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: _port_returns(x) - target_return},
        )
        result_optimization = sco.minimize(
            fun=_port_vol,
            x0=starting_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        frontier_returns.append(
            np.dot(result_optimization.x, mean_returns)
        )

        frontier_vols.append(
            np.sqrt(np.dot(result_optimization.x.T, np.dot(cov_matrix, result_optimization.x)))
        )

    return pd.DataFrame({
        'returns': frontier_returns,
        'volatility': frontier_vols
    })


def calc_excess_returns(
    df_returns: pd.DataFrame,
    df_market_data: pd.DataFrame,
    risk_free_col: str = 'DTB4WK',
    log_returns: bool = False,
) -> pd.DataFrame:
    """ 
    Calculate excess returns by subtracting risk free rate from asset returns.
    
    Notes:
    * Assumes "Date" as index in both dataframes.
    * Allows for log returns calculation if specified; be sure to provide risk_free_col in log 
      format if log_returns=True.
    """
    df_market_data = df_market_data[[risk_free_col]].copy()
    df_market_data.index = pd.to_datetime(df_market_data.index)
    df_returns.index = pd.to_datetime(df_returns.index)

    # Align market_data to the date range of returns
    start_date = df_returns.index.min()
    end_date = df_returns.index.max()
    
    # Ensure index is DatetimeIndex and selection is inclusive of end_date
    df_market_data = df_market_data.loc[start_date:end_date]
    if log_returns:
        df_returns = np.log(1 + df_returns)

    returns_excess = df_returns.apply(lambda x: x - df_market_data[risk_free_col], axis=0)

    return returns_excess


"""
Perform the Gibbons, Ross and Shanken (1989) Test
    - Gibbons, M., Ross, S., and Shanken, J., 1989, A Test of the Efficiency of A Given Portfolio,
      Econometrica 57, 1121-1152.
"""

def grs_test(resid: np.ndarray, alpha: np.ndarray, factors: np.ndarray) -> tuple:
    """ Perform the Gibbons, Ross and Shaken (1989) test.
        :param resid: Matrix of residuals from the OLS of size TxK.
        :param alpha: Vector of alphas from the OLS of size Kx1.
        :param factors: Matrix of factor returns of size TxJ.
        :return Test statistic and pValue of the test statistic.
    
    Note: slightly adjusted based on Cochrane (2005) notation.
    """
    # Determine the time series and assets
    iT, iN = resid.shape

    # Determine the amount of risk factors
    iK = factors.shape[1]

    # Input size checks
    assert alpha.shape == (iN, 1)
    assert factors.shape == (iT, iK)

    # Covariance of the residuals, variables are in columns.
    resid_cov = np.cov(resid, rowvar=False)

    # Mean of excess returns of the risk factors
    mu_f = np.nanmean(factors, axis=0)

    try:
        assert mu_f.shape == (1, iK)
    except AssertionError:
        mu_f = mu_f.reshape(1, iK)

    # Duplicate this series for T timestamps
    mu_f_extended = np.repeat(mu_f, iT, axis=0)

    # Test statistic
    mCovRF = (factors - mu_f_extended).T @ (factors - mu_f_extended) / (iT - 1)
    f_GRS = (iT / iN) * ((iT - iN - iK) / (iT - iK - 1)) * \
            (alpha.T @ (np.linalg.inv(resid_cov) @ alpha)) / \
            (1 + (mu_f @ (np.linalg.inv(mCovRF) @ mu_f.T)))

    pVal = 1 - f.cdf(f_GRS, iN, iT-iN-iK)
    return f_GRS[0][0], pVal[0][0]


def load_OECD_data(
        path_and_filename: str,
        path_and_filename_of_country_dict: str = './data/country_dict_ISO_3_to_2.csv',
#        iso_to_country: pd.DataFrame
) -> pd.DataFrame:
    """ Load and process OECD data from local excel files. """

    df = pd.read_csv(path_and_filename)
    iso_to_country = pd.read_csv(path_and_filename_of_country_dict)
    iso_to_country.columns = ['ISO_3', 'ISO_2']

    country_dict_ISO_3_to_2 = dict(zip(
        iso_to_country['ISO_3'], iso_to_country['ISO_2']
    ))

    df = pd.read_csv(path_and_filename)

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

    df = df.rename(columns={
        'REF_AREA': 'Country',
        'TIME_PERIOD': 'Date',
        'OBS_VALUE': 'CLI',
    }) 
    df['Date'] = pd.to_datetime(df['Date']) + pd.offsets.MonthEnd(0)

    return df

    
def generate_event_study_results(
        event_data: pd.DataFrame,
        trad_country_returns: pd.DataFrame,
        csm_returns: pd.DataFrame,
        countries_of_interest: list[str],
        get_largest_or_smallest_event: Literal['largest', 'smallest'],
        data_set_name: str,
        n_events: int = 1,
        t_event_window: int = 12,
) -> dict[str, dict]:
    """ 
    Find extreme events for a given dataset by country and match these with returns over a
    given window.

    Note on usage:
        Expects the event_data in wide format, with Date (monthly, end of month) as index and
        country data in the columns. Columns named in ISO-2 country code format.
    """
    
    settings_dict = {
        'event_data_name': data_set_name,
        'largest_or_smallest': get_largest_or_smallest_event,
        'n_events': n_events,
        't_event_window': t_event_window
    }
    
    event_data = event_data.loc[csm_returns.index]

    # slice t_event_window months from the df on both sides, so that event study has sufficient data
    event_data = event_data.iloc[t_event_window:(event_data.shape[0]-t_event_window), :].copy()

    # Find, per country, the top-n events
    event_dictionary: dict[list] = {}
    for country in [c for c in countries_of_interest if c in event_data.columns]:
    # for country in countries_of_interest:
        if get_largest_or_smallest_event == 'largest':
            top_events = event_data.nlargest(n_events, country).index.to_list()
        if get_largest_or_smallest_event == 'smallest':
            top_events = event_data.nsmallest(n_events, country).index.to_list()
        event_dictionary[country] = top_events

    event_returns = {}
    event_returns_relative = {}
    event_returns_absolute_trad = {}
    event_returns_absolute_csm = {}
    for country, event_list in event_dictionary.items():
        event_returns_relative_by_country = []
        event_returns_absolute_trad_by_country = []
        event_returns_absolute_csm_by_country = []
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
            event_relative_returns_cumulative_centered.name = f'event-{event}'

            combined_return_series = pd.concat([
                trad_return_series_cumulative_centered,
                csm_return_series_cumulative_centered,
            ], axis=1, keys=['trad_returns', 'csm_returns'])

            event_returns[f'{country}-event-{event}'] = combined_return_series
            event_returns_relative_by_country.append(event_relative_returns_cumulative_centered)
            
            event_returns_absolute_trad_by_country.append(trad_return_series_cumulative_centered)
            event_returns_absolute_csm_by_country.append(csm_return_series_cumulative_centered)
        
        # aggregate across multiple events per country, only relevant for n_events > 1
        event_returns_relative[country] = pd.concat(event_returns_relative_by_country, axis=1).mean(axis=1)
        event_returns_absolute_trad[country] = pd.concat(event_returns_absolute_trad_by_country, axis=1).mean(axis=1)
        event_returns_absolute_csm[country] = pd.concat(event_returns_absolute_csm_by_country, axis=1).mean(axis=1)

    results = {
        'settings': settings_dict,
        'events': event_dictionary,
        'event_returns_relative': event_returns_relative,
        'event_returns_absolute_trad': event_returns_absolute_trad,
        'event_returns_absolute_csm': event_returns_absolute_csm,
    }

    return results



def graph_events(
    results_dictionary: dict,
    graph_type: Literal['agg_rel_ret', 'agg_abs_ret_1plot', 'unagg_rel_ret_by_country', 'unagg_abs_ret_by_country'],
    save_fig: bool = True,
    output_path: str = 'output/run4_1-mimicking_addl'
) -> any:
    """ Create event study graphs based on output from generate_event_study_results() func """

    
    results_settings = results_dictionary['settings']
    
    if graph_type == 'agg_rel_ret':
        ### single plot for all effects combined; allows multiple events (n > 1)
        combined_country_events = pd.DataFrame(pd.concat(results_dictionary['event_returns_relative'], axis=1).mean(axis=1))

        fig, ax = plt.subplots(figsize=(8,8))

        ax.plot(combined_country_events.index, combined_country_events, label=combined_country_events.columns[0])
        ax.set_title(f'Event study results for measure {results_settings['event_data_name']}, n_events={results_settings['n_events']}')
        ax.set_ylabel('Cumulative returns (relative)')
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


    if graph_type == 'agg_abs_ret_1plot':
        combined_country_events_abs = pd.concat([
            pd.concat(results_dictionary['event_returns_absolute_trad'], axis=1).mean(axis=1),
            pd.concat(results_dictionary['event_returns_absolute_csm'], axis=1).mean(axis=1),
        ], axis=1, keys=['Trad country index', 'CSM index'])

        fig, ax = plt.subplots(figsize=(8,8))

        ax.plot(combined_country_events_abs.index, combined_country_events_abs['Trad country index'], label=combined_country_events_abs.columns[0])
        ax.plot(combined_country_events_abs.index, combined_country_events_abs['CSM index'], label=combined_country_events_abs.columns[1])
        ax.set_title(f'Event study results for measure {results_settings['event_data_name']}, n_events={results_settings['n_events']}')
        ax.set_ylabel('Cumulative returns (absolute)')
        ax.legend()
        ax.grid(True)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-')

        # Find the position of 't=0' and add vertical line there
        if 't=0' in combined_country_events_abs.index:
            t0_position = list(combined_country_events_abs.index).index('t=0')
            ax.axvline(x=t0_position, color='black', linewidth=1, linestyle='-')

        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    if graph_type == 'unagg_rel_ret_by_country':
        # Create a grid of subplots (adjust nrows/ncols as needed)
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(14, 14))
        axes = axes.flatten()  # Flatten to easily iterate

        for idx, (name, df) in enumerate(results_dictionary['event_returns_relative'].items()):
            df = pd.DataFrame(df)
            ax = axes[idx]
            ax.plot(df.index, df, label=df.columns[0])
            # ax.plot(df.index, df.iloc[:, 0], label=df.columns[0])
            # ax.plot(df.index, df.iloc[:, 1], label=df.columns[1])
            if results_settings['n_events'] == 1:
                ax.set_title(f'{results_settings['event_data_name']}, {name}, event: {results_dictionary['events'][name][0].strftime('%m/%Y')}')
            else:
                ax.set_title(f'{results_settings['event_data_name']}, {name}, n_events={results_settings['n_events']}')
            # ax.legend('')
            ax.grid(True)
            ax.set_ylabel('Relative return')

            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
            
            # Find the position of 't=0' and add vertical line there
            if 't=0' in df.index:
                t0_position = list(df.index).index('t=0')
                ax.axvline(x=t0_position, color='black', linewidth=1, linestyle='-')
        
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.tight_layout()

    if graph_type == 'unagg_abs_ret_by_country':
        raise NotImplementedError('graph type "unagg_abs_ret_by_country" is not yet implemented')

    if save_fig:
        file_name = f'fig_event_study_aggregate_{results_settings['event_data_name']}.png'
        fig.savefig(f'{output_path}/{file_name}', bbox_inches='tight')

    return plt


