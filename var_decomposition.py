""" Implementation of VAR model for Campbell/Shiller return decomposition. """

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

n_obs = 100

# Let copilot simulate some dummy data for testing and debugging:
np.random.seed(42)
sr = np.random.normal(loc=0.07/12, scale=0.15/np.sqrt(12), size=n_obs)
dy = np.random.normal(loc=0.02, scale=0.005, size=n_obs)
ys = np.random.normal(loc=0.015, scale=0.005, size=n_obs)

dummy_data = pd.DataFrame({
    "Stock_Returns": sr,
    "Dividend_Yield": dy,
    "Yield_Spread": ys,
})


def get_var_decomp(
        data: pd.DataFrame,        
) -> tuple:
    """ Returns the VAR model results and the cash flow and discount rate news. """
    k_factors = data.shape[1]  # Number of factors (columns) in the data
    model = VAR(data)
    results = model.fit(maxlags=1)
    coef_matrix = results.coefs[0]  # Extract coefficients
    rho = 0.95
    rhoGamma = rho * coef_matrix
    L = rhoGamma @ np.linalg.inv(np.identity(k_factors) - rhoGamma)
    # e1 = np.ones(k_factors).reshape(-1, 1) - Campbell et al's (2013) paper do not clearly define e1, but it seems to be a vector where the first element is 1 and the rest is 0, as in Campbell (!991)
    e1 = [1 if i == 0 else 0 for i in range(k_factors)]  
    e1 = np.array(e1).reshape(-1, 1)

    # Extracts the vector of unexplained variance (error term) from the VAR model:
    u = np.diag(results.sigma_u).reshape(-1, 1)

    cash_flow_news = []
    discount_rate_news = []
    for i in range(len(results.resid)):
        u = results.resid.iloc[i, :].values.reshape(-1, 1)

        cf_news = float(((e1.T + e1.T @ L) @ u)[0, 0])
        dr_news = float((e1.T @ L @ u)[0, 0])

        cash_flow_news.append(cf_news)
        discount_rate_news.append(dr_news)
    
    return cash_flow_news, discount_rate_news, results


get_var_decomp(dummy_data)
