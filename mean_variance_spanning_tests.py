""" Python script for performing mean variance spanning test """
import numpy as np
import pandas as pd
import pickle

import statsmodels.api as sm

data_path = 'data/index_data/'  

# load data
with open(data_path + 'run4-mimicking.pkl', 'rb') as f:
    results = pickle.load(f)

trad_country_returns = results["trad_country_returns"]
csm_returns = results["csm_returns"]

spanning_results = {}
X = sm.add_constant(trad_country_returns)
for country in csm_returns.columns:
    y = csm_returns[country]
    model = sm.OLS(y, X).fit()
    spanning_results[country] = model

for test_asset, model in spanning_results.items():
    print(f"Results for CSM index country: {test_asset}")
    print(model.summary())
    print("\n")

# Combined model:
y_combined = csm_returns.values.flatten()
X_combined = np.tile(
    trad_country_returns.values, 
    (csm_returns.shape[1], 1)
)
X_combined = sm.add_constant(X_combined)

combined_model = sm.OLS(y_combined, X_combined).fit()
print(combined_model.summary())

f_test_result = combined_model.f_test("const = 0")
print(f_test_result.summary())



# Let N+K be the number of assets in the combined portfolio (benchmark assets)
# N contains the returns of portfolio with traditional country indices (test assets)
# K contains the returns of portfolio with CSM indices
# We test the null hypothesis that the mean variance frontier of 
# the benchmark assets (N+K) is the same as that of the test assets (N)


