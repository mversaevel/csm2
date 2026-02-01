""" collection of functions """
import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy.stats import f


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
    log_returns: bool = False
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

    
