""" collection of functions """
import numpy as np
import pandas as pd
import scipy.optimize as sco

def create_efficient_frontier(
        mean_returns: np.ndarray, 
        cov_matrix: pd.DataFrame, 
        target_ret_normalizer: int = 1000
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

    for loop_return in range(150):
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
