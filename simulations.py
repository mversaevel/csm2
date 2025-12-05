""" Random selection of countries to explore diversification benefits in a portfolio context."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statistics import geometric_mean

import pickle


def sample_random_country(country_list: list, n: int) -> list:
    """ Randomly select n countries from the DataFrame df.
    
    Args:
        df (DataFrame): DataFrame containing returns by country. Expects wide format, meaning
                        each column represents a country and each row represents a time period.
        n (int): Number of countries to select.
    """

    selected_countries = np.random.choice(country_list, size=n, replace=False).tolist()
    return selected_countries


def calculate_filtered_returns(df: pd.DataFrame, selected_countries: list) -> pd.Series:
    """ Calculate the equal weighted returns across selected countries
    
    Args:
        df (DataFrame): DataFrame containing returns by country.
        selected_countries (list): List of countries to calculate returns for.
    
    Returns:
        DataFrame: Time series of aggregated returns for the selected countries.
    """

    df = df[selected_countries].copy()  # Filter the DataFrame for selected countries
    returns = df[selected_countries].mean(axis=1) 
    
    return returns


def calculate_statistics(returns: pd.Series) -> dict:
    """ Calculate statistics for the returns series.
    
    Args:
        returns (Series): Series of returns.
    
    Returns:
        dict: Dictionary containing geometric returns, standard deviation, and Sharpe ratio.
    """
    
    geo_return = geometric_mean(1 + returns) - 1
    std_dev = returns.std()
    sharpe_ratio = geo_return / std_dev if std_dev != 0 else np.nan
    
    return {
        'return_ann': geo_return,
        'std_dev': std_dev,
        'sharpe_ratio': sharpe_ratio
    }


def run_simulations(df: pd.DataFrame, n: int, runs_per_simulation: int) -> pd.DataFrame:
    """ Run multiple simulations to select random countries and calculate returns.
    
    Args:
        df (DataFrame): DataFrame containing returns by country.
        n (int): Number of countries to select in each simulation.
        num_simulations (int): Number of simulations to run.
    
    Returns:
        DataFrame: DataFrame containing statistics for each simulation.
    """
    
    country_list = df.columns.tolist()

    results = []
    for _ in range(runs_per_simulation):
        selected_countries = sample_random_country(country_list, n)
        returns = calculate_filtered_returns(df, selected_countries)
        stats = calculate_statistics(returns)
        results.append(stats)

    results_to_df = pd.DataFrame(results)

    return results_to_df


def run_all_simulations(
        input_data: dict,
        n: int | list[int],
        runs_per_simulation: int
):
    """ Run multiple simulations for different numbers of countries.
    
    Operates as a wrapper for `run_simulations` to handle multiple datasets and country selections.

    Args:
        input_data (dict): Dictionary with DataFrames as values and corresponding names as keys.
        n (int or list of ints): Number of countries to select in each simulation.
        runs_per_simulation (int): Number of simulations to run.
    
    Returns:
        dict: Dictionary with keys as number of countries and values as DataFrames with statistics.
    """
    
    if isinstance(n, int):
        n = [n]

    results = {}

    for name, data in input_data.items():  
        for num_countries in n:
            results[f"{name}-{num_countries}"] = run_simulations(data, num_countries, runs_per_simulation)
            print(f"Completed simulations for {name} with {num_countries} countries.")

    return results


def plot_sim_results(
        results_quantiles: pd.DataFrame, 
        metric: str,
        n_runs: int,
        list_of_quantiles: list | None = None,
        facet_plot: bool = False,
        use_convergence_type: bool = False,
) -> None:
    """ Plot metric of choice by number of countries for different datasets. 

    Use convergence_type=True to plot based on convergence to the limit; only affects labels, 
    data should be provided by the user accordingly, i.e. calculated separately.    
    """

    if metric not in results_quantiles.columns:
        raise ValueError(f"Metric '{metric}' not found in combined DataFrame columns.")

    if list_of_quantiles is None:
        list_of_quantiles = results_quantiles.index.unique().tolist()
    else:
        if not set(list_of_quantiles).issubset(results_quantiles.index.unique()):
            raise ValueError("Some quantiles in list_of_quantiles are not present in the DataFrame.")
        results_quantiles = results_quantiles[results_quantiles.index.isin(list_of_quantiles)]

    # Pivot the DataFrame for easier plotting
    plot_df = results_quantiles.reset_index().pivot_table(
        index='num_countries',
        columns=['dataset', 'index'],
        values=metric
    )

    colors = {
        'csm': 'tab:blue',
        'trad': 'tab:orange'
    }

    if facet_plot:
        datasets = results_quantiles['dataset'].unique()
        fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            linestyles = ['-', '--', '-.', ':']
            for i, quantile in enumerate(list_of_quantiles):
                linestyle = linestyles[i % len(linestyles)]
                ax.plot(
                    plot_df.index,
                    plot_df[(dataset, quantile)],
                    marker='o',
                    color=colors[dataset],
                    linestyle=linestyle,
                    label=f"{dataset} - {quantile:.3f}"
                )
            ax.set_xlabel('Number of Countries')
            
            ax.set_title(f'Metric: {metric}; Index type: {dataset}; Random runs per level: {n_runs}')
            ax.legend(title='Percentile')
        
        # if use_convergence_type:
        #     axes[0].set_ylabel(f'Convergence (% of limit of {metric})')
        # else:
        #     axes[0].set_ylabel(f'Metric: {metric}')

        ylabel = f'Convergence (% of limit of {metric})' if use_convergence_type else f'Metric: {metric}'
        axes[0].set_ylabel(ylabel)
        
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        for dataset in results_quantiles['dataset'].unique():
            linestyles = ['-', '--', '-.', ':']
            for i, quantile in enumerate(list_of_quantiles):
                linestyle = linestyles[i % len(linestyles)]
                ax.plot(
                    plot_df.index,
                    plot_df[(dataset, quantile)],
                    marker='o',
                    color=colors[dataset],
                    linestyle=linestyle,
                    label=f"{dataset} - {quantile:.3f}"
                )
        ax.set_xlabel('Number of Countries')
        # if use_convergence_type:
        #     ax.set_ylabel(f'Convergence (% of limit of {metric})')
        # else:
        #     ax.set_ylabel(f'Metric: {metric}')

        ylabel = f'Convergence (% of limit of {metric})' if use_convergence_type else f'Metric: {metric}'
        ax.set_ylabel(ylabel)

        ax.set_title(f'Metric: {metric} by Number of Countries and index type; Random runs per level: {n_runs}')
        ax.legend(title='Dataset - Percentile')
        plt.tight_layout()
        plt.show()


def get_convergence_normalized_results(
        results_quantiles: pd.DataFrame,
        n_countries: int = 16
) -> pd.DataFrame:
    """ Normalize results based on convergence values at maximum number of countries.
    
    Args:
        results_quantiles (DataFrame): DataFrame containing quantile results from simulations. 
            Effectively this is the output from the main script, can be obtained from reading the
            .csv file saved there.
        n_countries (int): Number of countries to consider as the convergence point; main script
            uses 16, but explicit > implicit.

    """
    

    convergence_values_trad = {
        "return_ann": results_quantiles[(results_quantiles["num_countries"] == n_countries) & (results_quantiles["dataset"] == "trad")]["return_ann"].to_list()[0],
        "std_dev": results_quantiles[(results_quantiles["num_countries"] == n_countries) & (results_quantiles["dataset"] == "trad")]["std_dev"].to_list()[0],
        "sharpe_ratio": results_quantiles[(results_quantiles["num_countries"] == n_countries) & (results_quantiles["dataset"] == "trad")]["sharpe_ratio"].to_list()[0],
    }

    convergence_values_csm = {
        "return_ann": results_quantiles[(results_quantiles["num_countries"] == n_countries) & (results_quantiles["dataset"] == "csm")]["return_ann"].to_list()[0],
        "std_dev": results_quantiles[(results_quantiles["num_countries"] == n_countries) & (results_quantiles["dataset"] == "csm")]["std_dev"].to_list()[0],
        "sharpe_ratio": results_quantiles[(results_quantiles["num_countries"] == n_countries) & (results_quantiles["dataset"] == "csm")]["sharpe_ratio"].to_list()[0],
    }

    results_quantiles_normalized_convergence = results_quantiles.copy()

    for metric in ["return_ann", "std_dev", "sharpe_ratio"]:
        trad_convergence_value = convergence_values_trad[metric]
        csm_convergence_value = convergence_values_csm[metric]
        results_quantiles_normalized_convergence.loc[results_quantiles_normalized_convergence["dataset"] == "trad", metric] /= trad_convergence_value
        results_quantiles_normalized_convergence.loc[results_quantiles_normalized_convergence["dataset"] == "csm", metric] /= csm_convergence_value

    return results_quantiles_normalized_convergence



if __name__ == "__main__":
    # This block is for running the script directly, if needed.
        
    data_path = 'data/index_data/'  

    # load data
    with open(data_path + 'run4-mimicking.pkl', 'rb') as f:
        results = pickle.load(f)

    trad_country_returns = results["trad_country_returns"]
    csm_returns = results["csm_returns"]

    
    # settings: 
    QUANTILES = [0.025, 0.5, 0.975]  # Percentiles to calculate
    LEVELS = [2, 3, 5, 7, 10, 13, 16]  # Number of countries to select in each simulation
    N_RUNS = 10000 # Number of runs per simulation


    full_results = run_all_simulations(
        input_data={
            "csm": csm_returns,
            "trad": trad_country_returns
        },
        n=LEVELS, 
        runs_per_simulation=N_RUNS
    )

    results_quantiles = pd.concat(
        [value.quantile(QUANTILES).assign(sim=key) for key, value in full_results.items()]
    )

    results_quantiles[['dataset', 'num_countries']] = results_quantiles['sim'].str.split('-', expand=True)
    results_quantiles['num_countries'] = results_quantiles['num_countries'].astype(int)

    results_quantiles.to_csv(f'data/simulations_results_quantiles_n_runs_{N_RUNS}.csv')

    plot_sim_results(results_quantiles, metric="std_dev", n_runs=N_RUNS, list_of_quantiles=QUANTILES, facet_plot=True)
