""" Implementation of VAR model for Campbell/Shiller return decomposition """

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


n_obs = 100
k_factors = 3  

# Let copilot simulate some dummy data
np.random.seed(42)
sr = np.random.normal(loc=0.07/12, scale=0.15/np.sqrt(12), size=n_obs)
dy = np.random.normal(loc=0.02, scale=0.005, size=n_obs)
ys = np.random.normal(loc=0.015, scale=0.005, size=n_obs)

dummy_data = pd.DataFrame({
    "Stock_Returns": sr,
    "Dividend_Yield": dy,
    "Yield_Spread": ys,
})

model = VAR(dummy_data)
results = model.fit(maxlags=1)

print("VAR Model Results:")
print(results.summary())

coef_matrix = results.coefs[0] # TODO - check if this extracts the correct coefficients, or if copilot is making stuff up
rho = 0.95
rhoGamma = rho * coef_matrix
L = rhoGamma @ np.linalg.inv(np.identity(k_factors) - rhoGamma)
e1 = np.ones(k_factors).reshape(-1, 1)

# Extracts the vector of unexplained variance (error term) from the VAR model:
u = np.diag(results.sigma_u).reshape(-1, 1)

cash_flow_news = (e1.T + e1.T @ L) @ u
discount_rate_news = e1.T @ L @ u


