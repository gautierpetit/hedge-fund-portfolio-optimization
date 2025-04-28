# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 00:02:43 2023

@author: gauti
"""

# Import modules:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from scipy.stats import genpareto, t
from statsmodels.distributions.copula.api import StudentTCopula
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from scipy.optimize import minimize
import seaborn as sns
import copy
from sklearn.linear_model import LinearRegression

# Custom files
import portfolios_functions as pf


###############################################################################


#                      Import and clean data                                  #


###############################################################################


Price = pd.read_excel(
    "Data/HFRI_full.xlsx", index_col=0, parse_dates=True, date_format="%b-%y"
)

Benchmark = pd.read_excel(
    "Data/Benchmark.xlsx", index_col=0, parse_dates=True, date_format="%m/%d/%Y"
)


Price_index = [
    "Event-Driven (Total)",
    "ED: Activist",
    "ED: Distressed/Restructuring",
    "ED: Merger Arbitrage",
    "Equity Hedge (Total)",
    "EH: Fundamental Value",
    "EH: Long/Short",
    "EH: Equity Market Neutral",
    "EH: Quantitative",
    "Fund of Funds Composite",
    "Macro (Total)",
    "Macro: Systematic",
    "Macro: Discretionary",
    "Relative Value (Total)",
    "RV: Asset Backed",
    "RV: Convertible Arbitrage",
]

Bench_index = [
    "S&P 500 COMPOSITE",
    "S&P US AGGREGATE BOND",
    "S&P GSCI Commodity",
    "Bloomberg U.S. Aggregate",
]


Price.columns = Price_index
Price.drop(index=Price[:3].index, inplace=True)
Price.index = pd.to_datetime(Price.index, format="%Y-%m-%d %H:%M:%S")
Price = Price.astype(float)


Benchmark.drop(index=Benchmark[:1].index, inplace=True)
Benchmark.columns = Bench_index
Benchmark = Benchmark.astype(float)
Benchmark.index = pd.to_datetime(Benchmark.index, format="%Y-%m-%d %H:%M:%S")
Benchmark = 100 * Benchmark / Benchmark.iloc[0]
Benchmark.drop(columns=["S&P US AGGREGATE BOND", "S&P GSCI Commodity"], inplace=True)

# Compute returns

Returns = Price / Price.shift(1) - 1
Returns.drop(index=Returns[:1].index, inplace=True)


Benchmark1 = pd.DataFrame(
    data=np.copy(Price["Fund of Funds Composite"]),
    index=Price.index,
    columns=["Fund of Funds Composite"],
)
Returns.drop(columns=["Fund of Funds Composite"], inplace=True)
Price.drop(columns=["Fund of Funds Composite"], inplace=True)

###############################################################################
###############################################################################
###############################################################################
###############################################################################


#                             Data analysis                                   #


###############################################################################
###############################################################################
###############################################################################
###############################################################################


# --- Annualized Mean Return ---
Returns_mean = (1 + Returns.mean()) ** (12) - 1


# --- Annualized Median Return ---
Returns_med = (1 + Returns.median()) ** (12) - 1


# --- Annualized Volatility ---
Returns_vola = Returns.std() * np.sqrt(12)


# --- Annualized Semi Volatility ---
Returns_semivola = Returns[Returns < 0].std() * np.sqrt(12)


# --- Minimum monthly return ---
Returns_min = Returns.min()


# --- Maximum monthly return---
Returns_max = Returns.max()


# --- Skewness ---
Returns_skew = Returns.skew()


# --- Excess Kurtosis ---
Returns_kurt = Returns.kurt() - 3


# ---- Correlation tables ----
Returns_corr = Returns.corr()

###############################################################################


#                     Autocorrelation Computation                             #


###############################################################################

# Compute autocorrelation for the first 5 order

Acf = pd.DataFrame(
    data=0,
    index=Returns.columns,
    columns=["ACF(1)", "ACF(2)", "ACF(3)", "ACF(4)", "ACF(5)"],
    dtype=float,
)

for x in Returns.columns:
    Acf.loc[x, "ACF(1)"] = Returns[x].autocorr(lag=1).copy()
    Acf.loc[x, "ACF(2)"] = Returns[x].autocorr(lag=2).copy()
    Acf.loc[x, "ACF(3)"] = Returns[x].autocorr(lag=3).copy()
    Acf.loc[x, "ACF(4)"] = Returns[x].autocorr(lag=4).copy()
    Acf.loc[x, "ACF(5)"] = Returns[x].autocorr(lag=5).copy()


###############################################################################


#                         Test for normality                                  #


###############################################################################

# Jarque and Bera test for normality

"""
# H0: Skewness = 0 and KKurtosis = 3
# We reject H0 if the JB statistic is greater than a chi-squared with 2 DF
# chi-squared(2) =5.99 at 95% significance
"""

n = len(Returns.columns)

JB_test = pd.DataFrame(
    data=0, index=Returns.columns, columns=["JB", "p value", "Result"], dtype=float
)

JB_test["JB"] = (n / 6) * (Returns.skew() ** 2 + (Returns.kurt() - 3) ** 2 / 4)

JB_test["p value"] = stats.distributions.chi2.sf(JB_test, 2)

# result of the test reject normality @ 5%, i.e if p value < 0.05 or 5.99 for
# Chi square(2)

JB_test["Result"] = JB_test["Result"].astype(str)

for x in JB_test.index:
    if JB_test.loc[x, "p value"] > 0.05:
        JB_test.loc[x, "Result"] = "normality"
    else:
        JB_test.loc[x, "Result"] = "reject"

del x, n

###############################################################################


#                        Test for autocorrelation                             #


###############################################################################

# Ljung-Box Q test for autocorrelation

"""
Ljung-Box Q test tests the null hypothesis that the autocorrelations up to the
specified lag are jointly zero.
if the p-value is less than a chosen significance level (e.g., 0.05 --> Chi
square 10 DOF = 18.307), it suggests evidence of autocorrelation in the data.
"""

LB_test = pd.DataFrame(
    data=0, index=Returns.columns, columns=["LB-Q", "p value", "Result"], dtype=float
)

for x in Returns.columns:
    LB_test.loc[x, "LB-Q"], LB_test.loc[x, "p value"] = acorr_ljungbox(
        Returns[x].dropna(), lags=[10]
    ).iloc[0]


#  result of the test shows autocorrelation @ 5%, i.e if p value < 0.05 or
# 18.307 for chi square(10)

LB_test["Result"] = LB_test["Result"].astype(str)

for x in LB_test.index:
    if LB_test.loc[x, "p value"] > 0.05:
        LB_test.loc[x, "Result"] = "reject"
    else:
        LB_test.loc[x, "Result"] = "auto correlation"

del x

###############################################################################


#                        Test for heteroskedasticity                          #


###############################################################################

# Engle LM (ARCH) test for heteroskedasticity

"""
The Engle LM (ARCH) test is a test for conditional heteroscedasticity in the
residual errors of a model. It tests the null hypothesis that there is no ARCH
effect up to the specified lag. If the p-value is below a chosen significance
level (e.g., 0.05), it suggests evidence of ARCH effects in the data.
"""

ARCH_test = pd.DataFrame(
    data=0, index=Returns.columns, columns=["ARCH", "p value", "Result"], dtype=float
)

for x in Returns.columns:
    # Create an ARCH(4) model
    model = arch_model(100 * Returns[x].dropna(), vol="ARCH", lags=4)

    # Fit the model
    model_fit = model.fit(disp="off")

    # Perform the Engle LM (ARCH) test
    lm_test = model_fit.arch_lm_test(lags=4)

    ARCH_test.loc[x, "ARCH"] = lm_test.stat
    ARCH_test.loc[x, "p value"] = lm_test.pval

    del lm_test, model, model_fit

# result of the test shows ARCH effect @ 5%, i.e if p value < 0.05 or 9.488 for
# chi square(4)

ARCH_test["Result"] = ARCH_test["Result"].astype(str)

for x in ARCH_test.index:
    if ARCH_test.loc[x, "p value"] > 0.05:
        ARCH_test.loc[x, "Result"] = "no effect"
    else:
        ARCH_test.loc[x, "Result"] = "ARCH effect"

del x

###############################################################################


#                            Statistics Table                                 #


###############################################################################

table = pd.concat(
    [
        Returns_mean,
        Returns_med,
        Returns_vola,
        Returns_semivola,
        Returns_min,
        Returns_max,
        Returns_skew,
        Returns_kurt,
        Acf,
    ],
    axis=1,
    ignore_index=True,
)
table.columns = [
    "Mean",
    "Median",
    "Volatility",
    "Semi Volatility",
    "Min. monthly",
    "Max. monthly",
    "Skewness",
    "Kurtosis",
    "ACF(1)",
    "ACF(2)",
    "ACF(3)",
    "ACF(4)",
    "ACF(5)",
]
del Returns_mean, Returns_med, Returns_vola, Returns_semivola, Returns_skew
del Returns_kurt, Returns_min, Returns_max, Acf


table_2 = pd.concat([JB_test, LB_test, ARCH_test], axis=1, ignore_index=True)
table_2.columns = [
    "JB",
    "p value",
    "Normality",
    "LB-Q",
    "p value",
    "Auto Correlation",
    "ARCH",
    "p value",
    "Heteroscedasticity",
]
del JB_test, LB_test, ARCH_test


with pd.ExcelWriter("Indexes summary.xlsx") as writer:
    table.to_excel(writer, sheet_name="Stats")
    table_2.to_excel(writer, sheet_name="Tests")
del writer
###############################################################################


#                           Data Plots                                        #


###############################################################################

Returns.plot(figsize=(16, 9))
plt.title("Monthly Returns of HFRI indices")
plt.legend(Returns.columns, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5)
plt.autoscale(tight="x")
plt.ylabel("Index returns")
plt.xlabel("Time")
plt.savefig("Returns.png")

Price.plot(figsize=(16, 9))
plt.autoscale(tight="x")
plt.ylabel("Index Price")
plt.xlabel("Time")
plt.ylim(0, 30000)
plt.legend(Price.columns, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5)
plt.title("Prices of of HFRI indices")
plt.savefig("Price.png")


###############################################################################
###############################################################################
###############################################################################
###############################################################################


#                               Methodology                                   #


###############################################################################
###############################################################################
###############################################################################
###############################################################################

###############################################################################


#                          Volatility Modelling                               #


###############################################################################

# AR(1)-EGARCH(1,1) model

AR_EGARCH_residuals = pd.DataFrame(columns=Returns.columns)
# AR_EGARCH_std_residuals = pd.DataFrame(columns=Returns.columns)
AR_EGARCH_vola = pd.DataFrame(columns=Returns.columns)

for col in Returns.columns:
    # Fit AR(1)-EGARCH(1,1) model to the returns
    model = arch_model(
        100 * Returns[col].dropna(),
        vol="EGarch",
        p=1,
        q=1,
        o=1,
        dist="t",
        mean="AR",
        lags=1,
    )
    results = model.fit(disp="off")

    # results.summary()

    # Obtain the residuals from the EGARCH model
    # Residuals from the model are standardized
    AR_EGARCH_residuals[col] = pd.Series(results.std_resid)

    # Obtain the dynamic volatility from the EGARCH model
    AR_EGARCH_vola[col] = pd.Series(results.conditional_volatility)

    del model, results

###############################################################################


#           Extreme value theory(POT using GPD, Gaussian Kernel)              #


###############################################################################

AR_EGARCH_residuals.drop(inplace=True, index=AR_EGARCH_residuals.index[0])


# Set the threshold for extreme events

upper_threshold = pd.Series(np.nan, index=Returns.columns)
lower_threshold = pd.Series(np.nan, index=Returns.columns)

for col in Returns.columns:
    # 95th percentile upper and lower threshold

    upper_threshold[col] = np.percentile(AR_EGARCH_residuals[col].dropna(), 95)

    lower_threshold[col] = np.percentile(AR_EGARCH_residuals[col].dropna(), 5)


# Apply Peaks Over Threshold method

upper_returns = {}
lower_returns = {}


for col in Returns.columns:
    # Upper return extremes
    extremes_upper = AR_EGARCH_residuals[col][
        AR_EGARCH_residuals[col] > upper_threshold[col]
    ]

    # Fit the Generalized Pareto Distribution (GPD) to upper exceedances
    params_upper = genpareto.fit(extremes_upper)

    """
    sns_plot = plt.figure(figsize=(16, 9))
    sns.lineplot(data=AR_EGARCH_residuals[col])
    plt.axhline(y=upper_threshold[col], linestyle='--', color='red')
    plt.legend(['Returns', 'Threshold'], loc="best")
    plt.title(col)
    plt.autoscale(tight='x')
    del sns_plot
    """
    # Replace extremes by extremes estimate from GPD
    upper_returns[col] = pd.Series(
        data=genpareto.rvs(*params_upper, size=len(extremes_upper), random_state=2),
        index=AR_EGARCH_residuals.index[
            np.where(AR_EGARCH_residuals[col] > upper_threshold[col])
        ],
    )

    del extremes_upper, params_upper

    # Lower return extremes
    extremes_lower = AR_EGARCH_residuals[col][
        AR_EGARCH_residuals[col] < lower_threshold[col]
    ]

    # Fit the Generalized Pareto Distribution (GPD) to lower exceedances
    params_lower = genpareto.fit(extremes_lower)

    """
    sns_plot = plt.figure(figsize=(16, 9))
    sns.lineplot(data=AR_EGARCH_residuals[col])
    plt.axhline(y=lower_threshold[col], linestyle='--', color='red')
    plt.legend(['Returns', 'Threshold'], loc="best")
    plt.title(col)
    plt.autoscale(tight='x')
    del sns_plot
    """
    # Replace extremes by extremes estimate from GPD
    lower_returns[col] = pd.Series(
        data=genpareto.rvs(*params_lower, size=len(extremes_lower), random_state=2),
        index=AR_EGARCH_residuals.index[
            np.where(AR_EGARCH_residuals[col] < lower_threshold[col])
        ],
    )

    del extremes_lower, params_lower


semipara_std_residuals = pd.DataFrame(
    data=np.copy(AR_EGARCH_residuals),
    index=AR_EGARCH_residuals.index,
    columns=AR_EGARCH_residuals.columns,
)

# Replace tails with parametrically estimated ones
for col in Returns.columns:

    semipara_std_residuals.loc[lower_returns[col].index, col] = lower_returns[col]
    semipara_std_residuals.loc[upper_returns[col].index, col] = upper_returns[col]

    """
    sns_plot = plt.figure(figsize=(16, 9))
    sns.lineplot(data=semipara_std_residuals[col])
    sns.lineplot(data=AR_EGARCH_residuals[col])
    plt.axhline(y=upper_threshold[col], linestyle='--', color='red')
    plt.axhline(y=lower_threshold[col], linestyle='--', color='red')
    plt.legend(['Extreme events', 'Returns', 'Threshold'], loc="best")
    plt.title(col)
    plt.autoscale(tight='x')
    del sns_plot
    """


# Gaussian kernel smoothing of center part of the distribution (middle 90%)
tails = {}
center = {}
smoothed_returns = {}


# FIXME: Parameter of Gaussian filter:
gaussian_std = 0.75


for col in Returns.columns:
    tails[col] = pd.concat(
        [lower_returns[col].index.to_series(), upper_returns[col].index.to_series()]
    )
    center[col] = AR_EGARCH_residuals[col].index.to_series().drop(index=tails[col])

    smoothed_returns[col] = pd.Series(
        gaussian_filter1d(AR_EGARCH_residuals[col].loc[center[col]], gaussian_std),
        index=center[col],
    )

    """
    # Plot effect of gaussian kernel

    sns_plot = plt.figure(figsize=(16, 9))
    sns.lineplot(data=AR_EGARCH_residuals[col].loc[center[col]])
    sns.lineplot(data=smoothed_returns[col])
    plt.legend(['Returns', 'Smoothed'], loc="best")
    plt.autoscale(tight='x')
    plt.title(col)
    plt.ylim(-2.5, 2.5)
    del sns_plot
    """


# Combine synthetic tails and smoothed center

for col in Returns.columns:

    semipara_std_residuals.loc[smoothed_returns[col].index, col] = smoothed_returns[col]
    # Plot residuals before and after semi-parametric approach
    """
    sns_plot = plt.figure(figsize=(16, 9))
    sns.lineplot(data=AR_EGARCH_residuals[col])
    sns.lineplot(data=semipara_std_residuals[col])
    plt.axhline(y=upper_threshold[col], linestyle='--', color='red')
    plt.axhline(y=lower_threshold[col], linestyle='--', color='red')
    plt.legend(['Original Returns', 'Smoothed center + extremes', 'Threshold'],
               loc="best")
    plt.title(col)
    plt.autoscale(tight='x')
    del sns_plot
    """


###############################################################################


#                               Copulas                                       #


###############################################################################


# FIXME: Parameters for Copulas
nu = 15  # Degrees of freedom


# Create a Student's t copula
t_dist = StudentTCopula(
    # Check that correlation matrix is positive semi definite
    # np.linalg.eigvals(semipara_std_residuals.corr())
    corr=semipara_std_residuals[25:].corr(),
    df=nu,
)

# Draw dependent uniform marginals from the Student's t copula

t_samples = pd.DataFrame(
    data=t_dist.rvs(len(semipara_std_residuals), random_state=42),  # 2,2,42
    columns=Returns.columns,
    index=semipara_std_residuals.index,
)


# Transform dependent uniform marginals into standardized residuals using student's t inverse CDF:

t_residuals = pd.DataFrame(
    data=t.ppf(t_samples, nu),
    columns=Returns.columns,
    index=semipara_std_residuals.index,
)


synthetic_returns = pd.DataFrame(
    columns=Returns.columns, index=semipara_std_residuals.index
)
# Reinject volatility from EGARCH model into returns
for col in Returns.columns:
    synthetic_returns[col] = (
        t_residuals[col] * (AR_EGARCH_vola[col][1:] / 100) + Returns[col].mean()
    )

    # Plot a comparison between original returns and semi-param synthetic returns

    """
    sns_plot = plt.figure(figsize=(16, 9))
    sns.lineplot(data=Returns[col])
    sns.lineplot(data=synthetic_returns[col])
    plt.legend(['Original Returns', 'Smoothed center + extremes'],
           loc="best")
    plt.title(col)
    plt.autoscale(tight='x')
    plt.ylim(-0.1,0.1)
    """


###############################################################################


#                          Create rolling windows                             #


###############################################################################


# Create DataFrame corresponding to our rolling window
rw = 12 * 4

rw_number = len(Returns) - rw

# Rolling window of returns
rw_returns = [0] * rw_number

for i in range(rw_number):
    rw_returns[i] = pd.DataFrame(
        data=Returns.iloc[i : i + rw],
        columns=Returns.columns,
        index=Returns.iloc[i : i + rw].index,
    )
    rw_returns[i].dropna(inplace=True, axis=1)


rw_corr_number = len(synthetic_returns) - rw

rw_corr_returns = [0] * rw_corr_number

for i in range(rw_corr_number):
    rw_corr_returns[i] = pd.DataFrame(
        data=synthetic_returns.iloc[i : i + rw],
        columns=synthetic_returns.columns,
        index=synthetic_returns.iloc[i : i + rw].index,
    )

    rw_corr_returns[i].dropna(axis=1, inplace=True)


# Rolling window of correlation for both S&P and bond
# Get the data necessary for the correlation to stocks
bench_corr = pd.read_excel(
    "Data/Benchmark.xlsx", sheet_name="Correlation", index_col=0, parse_dates=True
)

bench_corr.index = pd.to_datetime(bench_corr.index, format="%Y-%m-%d")
bench_corr = bench_corr.astype(float)
bench_corr.drop(index=bench_corr[-1:].index, inplace=True)

bench_returns = bench_corr / bench_corr.shift(1) - 1
bench_returns.drop(index=bench_returns[:1].index, inplace=True)


rw_sp = [0] * rw_number
rw_bond = [0] * rw_number
# Compute the correlation for each column in a rolling window form
for i in range(rw_number):
    rw_sp[i] = pd.Series(
        data=[
            pd.concat(
                [
                    Returns[col].iloc[i : i + rw],
                    bench_returns["SPX - Adj Close"].iloc[i : i + rw],
                ],
                axis=1,
                ignore_index=False,
            )
            .corr()
            .iloc[0][1]
            for col in Returns.columns
        ],
        index=Returns.columns,
        name=Returns.iloc[i + rw].name,
    )
    rw_bond[i] = pd.Series(
        data=[
            pd.concat(
                [
                    Returns[col].iloc[i : i + rw],
                    bench_returns[" ICE BofA US Corporate Index Total Return"].iloc[
                        i : i + rw
                    ],
                ],
                axis=1,
                ignore_index=False,
            )
            .corr()
            .iloc[0][1]
            for col in Returns.columns
        ],
        index=Returns.columns,
        name=Returns.iloc[i + rw].name,
    )
del i


for i in range(rw_number):
    rw_sp[i].dropna(inplace=True)
    for col in rw_sp[i].index:
        if col not in rw_returns[i].columns:
            rw_sp[i].drop(index=col, inplace=True)

    rw_bond[i].dropna(inplace=True)
    for col in rw_bond[i].index:
        if col not in rw_returns[i].columns:
            rw_bond[i].drop(index=col, inplace=True)
del i


rw_corr_sp = copy.deepcopy(rw_sp.copy()[1:])
rw_corr_bond = copy.deepcopy(rw_bond.copy()[1:])

# One month offset in synthetic returns, means columns in rolling windows do not match for a few periods, setting correct columns for rw_corr that will be used in synthetic returns + correlation


for i in range(rw_corr_number):
    rw_corr_sp[i].dropna(inplace=True)
    for col in rw_corr_sp[i].index:
        if col not in rw_corr_returns[i].columns:
            rw_corr_sp[i].drop(index=col, inplace=True)

    rw_corr_bond[i].dropna(inplace=True)
    for col in rw_corr_bond[i].index:
        if col not in rw_corr_returns[i].columns:
            rw_corr_bond[i].drop(index=col, inplace=True)

del i


###############################################################################


#                           Function definition                               #


###############################################################################

# Definition of functions:


def portfolio_aum(weight):
    """
    Computes portfolio value based on Returns Dataframe
    """

    aum = pd.Series(
        data=[100] * (len(weight) + 1),
        index=Returns.iloc[rw - 1 :].index,
        name="AUM",
        dtype=float,
    )

    for i, w in enumerate(weight):

        columns_idx = Returns.columns.get_indexer(rw_returns[i].columns)

        returns_row = Returns.iloc[rw + i, columns_idx]

        aum.iloc[i + 1] = (1 + w.T @ returns_row) * aum.iloc[i]

    return aum


def portfolio_return(weight):
    """
    Computes portfolio returns based on Returns Dataframe
    """

    returns = pd.Series(
        data=[0] * (len(weight) + 1),
        index=Returns.iloc[rw - 1 :].index,
        name="Return",
        dtype=float,
    )

    for i, w in enumerate(weight):

        columns_idx = Returns.columns.get_indexer(rw_returns[i].columns)

        returns_row = Returns.iloc[rw + i, columns_idx]

        returns.iloc[i + 1] = w.T @ returns_row

    return returns


def turnover(weight):
    """
    Computes portfolio turnover with absolute change in weights
    """
    to = pd.Series(
        data=[0] * len(weight),
        index=Returns.iloc[rw:].index,
        name="Turnover",
        dtype=float,
    )

    for i in range(len(weight) - 1):

        columns_idx = Returns.columns.get_indexer(rw_returns[i].columns)

        returns_row = Returns.iloc[rw + i, columns_idx]

        w1 = weight[i + 1].squeeze().values
        w0 = (weight[i].squeeze() * (1 + returns_row)).values

        length_diff = len(w1) - len(w0)

        if length_diff > 0:
            w0 = np.pad(w0, (0, length_diff), mode="constant")

        to.iloc[i + 1] = abs(w1 - w0).sum()

    return to


def portfolio_aum_tc(weight, to, tc):
    """
    Computes portfolio value with trading costs based on Returns Dataframe
    """

    aum_tc = pd.Series(
        data=[100] * (len(weight) + 1),
        index=Returns.iloc[rw - 1 :].index,
        name="AUM",
        dtype=float,
    )

    for i, w in enumerate(weight):

        columns_idx = Returns.columns.get_indexer(rw_returns[i].columns)

        returns_row = Returns.iloc[rw + i, columns_idx]

        net_return = (w.T @ returns_row) - to.iloc[i] * tc

        aum_tc.iloc[i + 1] = (1 + net_return) * aum_tc.iloc[i]

    return aum_tc


def portfolio_return_tc(weight, to, tc):
    """
    Computes portfolio returns based on Returns Dataframe
    """

    returns = pd.Series(
        data=[0] * (len(weight) + 1),
        index=Returns.iloc[rw - 1 :].index,
        name="Return",
        dtype=float,
    )

    for i, w in enumerate(weight):

        columns_idx = Returns.columns.get_indexer(rw_returns[i].columns)

        returns_row = Returns.iloc[rw + i, columns_idx]

        returns.iloc[i + 1] = (w.T @ returns_row) - to.iloc[i] * tc

    return returns


def portfolio_correlation(weight, corr):
    """
    Computes the correlation of a portfolio based on the correlation of its individual assets
    """

    correl = pd.Series(
        data=[np.nan] * rw_number,
        index=Returns.iloc[rw : rw + rw_number].index,
        name="Correlation",
        dtype=float,
    )

    for i in range(rw_number):
        correl.iloc[i] = weight[i].mul(corr[i], axis=0).sum()

    return correl


def portfolio_syn_correlation(weight, corr):
    """
    Computes the synthetic correlation of a portfolio based on the correlation of its individual assets
    """

    correl = pd.Series(
        data=[np.nan] * rw_corr_number,
        index=Returns.iloc[rw + 1 : rw + rw_corr_number + 1].index,
        name="Syn_Correlation",
        dtype=float,
    )

    for i in range(rw_corr_number):
        correl.iloc[i] = weight[i].mul(corr[i], axis=0).sum()

    return correl


# FIXME: Default transaction cost 30bps
def portfolio(weight, tc=0.003):
    """
    Outputs portfolio value, return, turnover, value with transaction cost, returns with transaction costs, correlation of the portfolio to S&P500, correlation of the portfolio to bond index. Output is individual Series.
    """
    aum = portfolio_aum(weight)

    returns = portfolio_return(weight)

    to = turnover(weight)

    aum_tc = portfolio_aum_tc(weight, to, tc)

    returns_tc = portfolio_return_tc(weight, to, tc)

    correl_sp = portfolio_correlation(weight, rw_sp)
    correl_bond = portfolio_correlation(weight, rw_bond)

    return aum, returns, to, aum_tc, returns_tc, correl_sp, correl_bond


# FIXME: Default transaction cost 30bps
def portfolio_syn(weight_syn, tc=0.003):
    """
    Outputs portfolio value, return, turnover, value with trading cost, returns with trading costs, correlation of the portfolio to S&P500, correlation of the portfolio to bond index. Output is individual Series.
    """
    aum = portfolio_aum(weight_syn)

    returns = portfolio_return(weight_syn)

    to = turnover(weight_syn)

    aum_tc = portfolio_aum_tc(weight_syn, to, tc)

    returns_tc = portfolio_return_tc(weight_syn, to, tc)

    correl_sp = portfolio_syn_correlation(weight_syn, rw_corr_sp)
    correl_bond = portfolio_syn_correlation(weight_syn, rw_corr_bond)

    return aum, returns, to, aum_tc, returns_tc, correl_sp, correl_bond


###############################################################################

# Graphical function usefull  to observe the impact of cost minimization on turnover, the constraint on correlation and the weight allocation

###############################################################################


def turnover_plot(to_mint, to):
    """
    Outputs a graph plotting comparison between portfolios with and without costs minimization
    """
    var_name = str([name for name, value in globals().items() if value is to_mint][0])

    # Create a new figure and axes for each plot
    fig, ax = plt.subplots(figsize=(16, 9))

    to.plot(
        kind="area",
        legend=None,
        ax=ax,
        color="tab:blue",
        label="No minimization: Total To: " + str(round(to.sum(), 2)),
    )
    to_mint.plot(
        kind="area",
        ax=ax,
        color="tab:red",
        label="With minimization: Total To: " + str(round(to_mint.sum(), 2)),
    )
    plt.title(
        "Turnover comparison between portfolios with and without costs minimization "
    )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
    )
    plt.autoscale(tight="x")
    plt.ylim(0, 2)
    plt.ylabel("Turnover")
    plt.xlabel("Time")
    plt.savefig("Turnover/" + var_name + "_costs.png")


def correlation_plot(corrsp_cons, corrbond_cons, corrsp, corrbond):
    """
    Outputs a graph plotting correlation of a portfolio to the S&P and bond index overtime
    """
    var_name = str([name for name, value in globals().items() if value is corrsp][0])

    # Create a new figure and axes for each plot
    fig, ax = plt.subplots(figsize=(16, 9))

    corrsp_cons.plot(legend=None, ax=ax, color="tab:blue", linestyle="--")
    corrbond_cons.plot(ax=ax, color="tab:red", linestyle="--")
    corrsp.plot(ax=ax, color="tab:blue", linestyle="dotted")
    corrbond.plot(ax=ax, color="tab:red", linestyle="dotted")

    plt.title("Portoflio correlation to the stock and bond index over time")
    plt.legend(
        [
            "S&P constrained",
            "Bond constrained",
            "S&P unconstrained",
            "Bond unconstrained",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
    )
    plt.autoscale(tight="x")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.ylim(-1, 1)
    plt.savefig("Correlation/" + var_name + "_correlations.png")


def stackplts(weights):
    data = pd.DataFrame(columns=Returns.columns, dtype=float)
    var_name = str([name for name, value in globals().items() if value is weights][0])

    for i in range(len(weights)):
        data = pd.concat([data, weights[i].T])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.stackplot(
        data.index,
        [data[col].fillna(0) for col in Returns.columns],
        labels=Returns.columns,
        colors=sns.color_palette("pastel", 15),
        baseline="zero",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=5,
    )
    ax.set_ylabel("Percentage ")
    ax.autoscale(tight="x")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)
    ax.set_title("Weight allocation of the portfolio over time")
    fig.savefig("Stackplots/" + var_name + ".png")


###############################################################################


#   Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity     #


###############################################################################

Riskfree = pd.read_excel(
    "Data/Benchmark.xlsx",
    sheet_name="Tbill 10y",
    index_col=0,
    parse_dates=True,
    date_format="%Y-%m-%d",
)

Riskfree.drop(index=Riskfree[0:10].index, inplace=True)
Riskfree.columns = ["DGS10"]
Riskfree = Riskfree.astype(float)
Riskfree.index = pd.to_datetime(Riskfree.index, format="%Y-%m-%d %H:%M:%S")
Riskfree = Riskfree / 100
Rf_mean = Riskfree.mean().iloc[0]
# we get the average monthly risk free rate from the average annualized 10y T-bill rate

rf = (Rf_mean + 1) ** (1 / 12) - 1

###############################################################################
###############################################################################

#                    Functions for performance computation                    #

###############################################################################


bench = (
    Benchmark1["Fund of Funds Composite"]
    / Benchmark1["Fund of Funds Composite"].shift(1)
    - 1
)
returns_bench = np.array(bench.iloc[rw:])
returns_syn_bench = np.array(bench.iloc[rw + 1 :])

aum_FoF = (
    Benchmark1["Fund of Funds Composite"].iloc[rw:]
    / Benchmark1["Fund of Funds Composite"].iloc[rw]
    * 100
)


def information_ratio(returns, benchmark_returns):
    """
    Calculate the Information Ratio of returns relative to a benchmark.

    """

    excess_returns = returns - benchmark_returns
    average_excess_return = np.mean(excess_returns)
    tracking_error = np.std(excess_returns, ddof=1)

    information_ratio = average_excess_return / tracking_error

    return information_ratio


def max_drawdown(returns):
    """
    Compute the maximum drawdown of returns.

    """

    n = len(returns)
    peak = returns.iloc[0]
    max_drawdown = 0.0

    for i in range(1, n):
        if returns.iloc[i] > peak:
            peak = returns.iloc[i]
        else:
            drawdown = (peak - returns.iloc[i]) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown


def r_squared(returns, index_returns):
    """
    Calculate the R-squared of returns with respect to an index.
    """
    # Reshape input data if necessary
    returns = np.array(returns).reshape(-1, 1)

    index_returns = np.array(index_returns).reshape(-1, 1)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(index_returns, returns)

    # Calculate R-squared value
    r_squared = model.score(index_returns, returns)

    return r_squared


def performance_measures(weight, aum, returns, to, syn=False):
    """
    Computes performance measures and creates a pandas Series

    """

    AR = (1 + returns.mean()) ** (12) - 1
    SD = returns.std() * np.sqrt(12)
    MDD = max_drawdown(aum)
    CVaR = pf.cvar(returns, 0.01)
    CDaR = pf.cdar(returns, 0.01)
    SR = (AR - Rf_mean) / SD
    M_squared = Rf_mean + SR * bench.iloc[rw:].std() * np.sqrt(12)
    Calmar = (AR - Rf_mean) / MDD

    if syn == True:
        R_squared = r_squared(returns, returns_syn_bench)
        CORR_SP = portfolio_syn_correlation(weight, rw_corr_sp).mean()
        CORR_BOND = portfolio_syn_correlation(weight, rw_corr_bond).mean()
        CORR_bench = returns.corr(bench.iloc[rw + 1 :])
    else:
        R_squared = r_squared(returns, returns_bench)
        CORR_SP = portfolio_correlation(weight, rw_sp).mean()
        CORR_BOND = portfolio_correlation(weight, rw_bond).mean()
        CORR_bench = returns.corr(bench.iloc[rw:])
    PT = to.sum()

    table = pd.Series(
        data=[
            AR,
            M_squared,
            SD,
            MDD,
            CVaR,
            CDaR,
            SR,
            Calmar,
            R_squared,
            CORR_SP,
            CORR_BOND,
            CORR_bench,
            PT,
        ],
        index=[
            "Return",
            "M squared",
            "Volatility",
            "MDD",
            "CVaR",
            "CDaR",
            "Sharpe",
            "Calmar",
            "R squared",
            "Corr. Stocks",
            "Corr. Bonds",
            "Corr. FoF",
            "Turnover",
        ],
    )

    return table



###############################################################################
###############################################################################
###############################################################################


#                           Bechmark Portfolios                               #


###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Weight here equal 0, used to get the performance table,
weight_FoF = [0] * rw_number

for i in range(rw_number):
    weight_FoF[i] = pd.DataFrame(
        data=[0] * len(rw_returns[i].T),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )


t_FoF = performance_measures(weight_FoF, aum_FoF, bench.iloc[rw:], np.array(0), False).T
t_FoF["Corr. Stocks"] = (
    pd.concat([bench.iloc[rw:], bench_returns["SPX - Adj Close"].iloc[rw:]], axis=1)
    .corr()
    .iloc[0][1]
)
t_FoF["Corr. Bonds"] = (
    pd.concat(
        [
            bench.iloc[rw:],
            bench_returns[" ICE BofA US Corporate Index Total Return"].iloc[rw:],
        ],
        axis=1,
    )
    .corr()
    .iloc[0][1]
)
t_FoF = pd.DataFrame(t_FoF, columns=["FoF"])

# Equally weighted portfolio

weight_EW = [0] * rw_number

for i in tqdm(range(rw_number), desc="EW optimization"):
    weight_EW[i] = pd.DataFrame(
        data=[1 / len(rw_returns[i].T)] * len(rw_returns[i].T),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )


del i


aum_EW, returns_EW, to_EW, aum_EW_tc, returns_EW_tc, corrsp_EW, corrbond_EW = portfolio(
    weight_EW
)

stackplts(weight_EW)

t_EW = pd.concat(
    [
        performance_measures(weight_EW, aum_EW, returns_EW, to_EW, False),
        performance_measures(weight_EW, aum_EW_tc, returns_EW_tc, to_EW, False),
    ],
    axis=1,
    ignore_index=True,
)
t_EW.columns = ["EW - No costs", "EW - With costs"]

###############################################################################

# Minimum Variance Portfolio

weight_MVP = [0] * rw_number

for i in tqdm(range(rw_number), desc="MVP optimization"):
    weight_MVP[i] = pd.DataFrame(
        data=pf.MVP(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_MVP,
    returns_MVP,
    to_MVP,
    aum_MVP_tc,
    returns_MVP_tc,
    corrsp_MVP,
    corrbond_MVP,
) = portfolio(weight_MVP)

stackplts(weight_MVP)


t_MVP = pd.concat(
    [
        performance_measures(weight_MVP, aum_MVP, returns_MVP, to_MVP, False),
        performance_measures(weight_MVP, aum_MVP_tc, returns_MVP_tc, to_MVP, False),
    ],
    axis=1,
    ignore_index=True,
)
t_MVP.columns = ["MVP - No costs", "MVP - With costs"]

###############################################################################

# Maximum Sharpe ratio portfolio

weight_MSR = [0] * rw_number

for i in tqdm(range(rw_number), desc="MSR optimization"):
    weight_MSR[i] = pd.DataFrame(
        data=pf.MV_risk(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_MSR,
    returns_MSR,
    to_MSR,
    aum_MSR_tc,
    returns_MSR_tc,
    corrsp_MSR,
    corrbond_MSR,
) = portfolio(weight_MSR)

stackplts(weight_MSR)

t_MSR = pd.concat(
    [
        performance_measures(weight_MSR, aum_MSR, returns_MSR, to_MSR, False),
        performance_measures(weight_MSR, aum_MSR_tc, returns_MSR_tc, to_MSR, False),
    ],
    axis=1,
    ignore_index=True,
)
t_MSR.columns = ["MSR - No costs", "MSR - With costs"]


###############################################################################


t_bench = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
    ],
    axis=1,
    ignore_index=False,
)
with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_bench.to_excel(writer, sheet_name="Benchmark")


sns_plot = plt.figure(figsize=(16, 9))
plt.title("Benchmark Portfolio Wealth (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted - no costs",
        "Equally Weighted - with costs",
        "Minimum Variance - no costs",
        "Minimum Variance - with costs",
        "Maximum Sharpe ratio - no costs",
        "Maximum Sharpe ratio - with costs",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=4,
)
plt.savefig("Benchmark.png")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#                Portfolio computation is in multiple parts:


# Part 1: Porfolios without additional constraints

# Part 2: Cost minimization added in objective function

# Part 3: Portfolio correlation constraints on stocks and bonds added

# Part 4: Both cost minimization and correlation constraints


# Minimum risk portfolios are minimum risk portfolios
# Optimal portfolios are optimal portfolios

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#              Part 1: Porfolios without additional constraints               #


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Minimum risk CVAR:

weight_CVAR = [0] * rw_number

for i in tqdm(range(rw_number), desc="CVAR optimization"):
    weight_CVAR[i] = pd.DataFrame(
        data=pf.CVAR(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CVAR,
    returns_CVAR,
    to_CVAR,
    aum_CVAR_tc,
    returns_CVAR_tc,
    corrsp_CVAR,
    corrbond_CVAR,
) = portfolio(weight_CVAR)

stackplts(weight_CVAR)

t_CVAR = pd.concat(
    [
        performance_measures(weight_CVAR, aum_CVAR, returns_CVAR, to_CVAR, False),
        performance_measures(weight_CVAR, aum_CVAR_tc, returns_CVAR_tc, to_CVAR, False),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR.columns = ["CVAR - No costs", "CVAR - With costs"]


# Optimal CVAR:

weight_CVAR_risk = [0] * rw_number

for i in tqdm(range(rw_number), desc="CVAR risk optimization"):
    weight_CVAR_risk[i] = pd.DataFrame(
        data=pf.CVAR_risk(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CVAR_risk,
    returns_CVAR_risk,
    to_CVAR_risk,
    aum_CVAR_risk_tc,
    returns_CVAR_risk_tc,
    corrsp_CVAR_risk,
    corrbond_CVAR_risk,
) = portfolio(weight_CVAR_risk)

stackplts(weight_CVAR_risk)


t_CVAR_risk = pd.concat(
    [
        performance_measures(
            weight_CVAR_risk, aum_CVAR_risk, returns_CVAR_risk, to_CVAR_risk, False
        ),
        performance_measures(
            weight_CVAR_risk,
            aum_CVAR_risk_tc,
            returns_CVAR_risk_tc,
            to_CVAR_risk,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_risk.columns = ["CVAR risk - No costs", "CVAR risk - With costs"]


###############################################################################

# Minimum risk CDAR:

weight_CDAR = [0] * rw_number

for i in tqdm(range(rw_number), desc="CDAR optimization"):
    weight_CDAR[i] = pd.DataFrame(
        data=pf.CDAR(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CDAR,
    returns_CDAR,
    to_CDAR,
    aum_CDAR_tc,
    returns_CDAR_tc,
    corrsp_CDAR,
    corrbond_CDAR,
) = portfolio(weight_CDAR)

stackplts(weight_CDAR)

t_CDAR = pd.concat(
    [
        performance_measures(weight_CDAR, aum_CDAR, returns_CDAR, to_CDAR, False),
        performance_measures(weight_CDAR, aum_CDAR_tc, returns_CDAR_tc, to_CDAR, False),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR.columns = ["CDAR - No costs", "CDAR - With costs"]


# Optimal CDAR:

weight_CDAR_risk = [0] * rw_number

for i in tqdm(range(rw_number), desc="CDAR risk optimization"):
    weight_CDAR_risk[i] = pd.DataFrame(
        data=pf.CDAR_risk(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CDAR_risk,
    returns_CDAR_risk,
    to_CDAR_risk,
    aum_CDAR_risk_tc,
    returns_CDAR_risk_tc,
    corrsp_CDAR_risk,
    corrbond_CDAR_risk,
) = portfolio(weight_CDAR_risk)

stackplts(weight_CDAR_risk)

t_CDAR_risk = pd.concat(
    [
        performance_measures(
            weight_CDAR_risk, aum_CDAR_risk, returns_CDAR_risk, to_CDAR_risk, False
        ),
        performance_measures(
            weight_CDAR_risk,
            aum_CDAR_risk_tc,
            returns_CDAR_risk_tc,
            to_CDAR_risk,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_risk.columns = ["CDAR risk - No costs", "CDAR risk - With costs"]


###############################################################################


# Minimum risk Omega:

weight_Omegamin = [0] * rw_number

for i in tqdm(range(rw_number), desc="Omegamin optimization"):
    weight_Omegamin[i] = pd.DataFrame(
        data=pf.Omega_min(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_Omegamin,
    returns_Omegamin,
    to_Omegamin,
    aum_Omegamin_tc,
    returns_Omegamin_tc,
    corrsp_Omegamin,
    corrbond_Omegamin,
) = portfolio(weight_Omegamin)

stackplts(weight_Omegamin)

t_Omegamin = pd.concat(
    [
        performance_measures(
            weight_Omegamin, aum_Omegamin, returns_Omegamin, to_Omegamin, False
        ),
        performance_measures(
            weight_Omegamin, aum_Omegamin_tc, returns_Omegamin_tc, to_Omegamin, False
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamin.columns = ["Omega denum. - No costs", "Omega denum. - With costs"]


# Optimal Omega:

weight_Omegamax = [0] * rw_number

for i in tqdm(range(rw_number), desc="Omegamax optimization"):
    weight_Omegamax[i] = pd.DataFrame(
        data=pf.Omega_max(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_Omegamax,
    returns_Omegamax,
    to_Omegamax,
    aum_Omegamax_tc,
    returns_Omegamax_tc,
    corrsp_Omegamax,
    corrbond_Omegamax,
) = portfolio(weight_Omegamax)

stackplts(weight_Omegamax)

t_Omegamax = pd.concat(
    [
        performance_measures(
            weight_Omegamax, aum_Omegamax, returns_Omegamax, to_Omegamax, False
        ),
        performance_measures(
            weight_Omegamax, aum_Omegamax_tc, returns_Omegamax_tc, to_Omegamax, False
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamax.columns = ["Omega ratio - No costs", "Omega ratio - With costs"]


###############################################################################

# Performance table

t_historical_P1 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_CVAR,
        t_CVAR_risk,
        t_CDAR,
        t_CDAR_risk,
        t_Omegamin,
        t_Omegamax,
    ],
    axis=1,
    ignore_index=False,
)
t_historical_P1.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_historical_P1.to_excel(writer, sheet_name="P1 - Historical")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using historical returns with costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_CVAR_tc)
sns.lineplot(data=aum_CVAR_risk_tc)
sns.lineplot(data=aum_CDAR_tc)
sns.lineplot(data=aum_CDAR_risk_tc)
sns.lineplot(data=aum_Omegamin_tc)
sns.lineplot(data=aum_Omegamax_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P1.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using historical returns without costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_CVAR)
sns.lineplot(data=aum_CVAR_risk)
sns.lineplot(data=aum_CDAR)
sns.lineplot(data=aum_CDAR_risk)
sns.lineplot(data=aum_Omegamin)
sns.lineplot(data=aum_Omegamax)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P1.png")


###############################################################################


#                    Optimization with synthetic returns                      #


###############################################################################


# Minimum risk CVAR

weight_syn_CVAR = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CVAR optimization"):
    weight_syn_CVAR[i] = pd.DataFrame(
        data=pf.CVAR(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CVAR,
    returns_syn_CVAR,
    to_syn_CVAR,
    aum_syn_CVAR_tc,
    returns_syn_CVAR_tc,
    corrsp_syn_CVAR,
    corrbond_syn_CVAR,
) = portfolio_syn(weight_syn_CVAR)

stackplts(weight_syn_CVAR)

t_syn_CVAR = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR, aum_syn_CVAR, returns_syn_CVAR, to_syn_CVAR, True
        ),
        performance_measures(
            weight_syn_CVAR, aum_syn_CVAR_tc, returns_syn_CVAR_tc, to_syn_CVAR, True
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR.columns = ["CVAR synth. - No costs", "CVAR synth. - With costs"]


# Optimal CVAR:

weight_syn_CVAR_risk = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CVAR risk optimization"):
    weight_syn_CVAR_risk[i] = pd.DataFrame(
        data=pf.CVAR_risk(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CVAR_risk,
    returns_syn_CVAR_risk,
    to_syn_CVAR_risk,
    aum_syn_CVAR_risk_tc,
    returns_syn_CVAR_risk_tc,
    corrsp_syn_CVAR_risk,
    corrbond_syn_CVAR_risk,
) = portfolio_syn(weight_syn_CVAR_risk)


stackplts(weight_syn_CVAR_risk)

t_syn_CVAR_risk = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_risk,
            aum_syn_CVAR_risk,
            returns_syn_CVAR_risk,
            to_syn_CVAR_risk,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_risk,
            aum_syn_CVAR_risk_tc,
            returns_syn_CVAR_risk_tc,
            to_syn_CVAR_risk,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_risk.columns = [
    "CVAR risk synth. - No costs",
    "CVAR risk synth. - With costs",
]


###############################################################################


# Minimum risk CDAR:

weight_syn_CDAR = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CDAR optimization"):
    weight_syn_CDAR[i] = pd.DataFrame(
        data=pf.CDAR(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CDAR,
    returns_syn_CDAR,
    to_syn_CDAR,
    aum_syn_CDAR_tc,
    returns_syn_CDAR_tc,
    corrsp_syn_CDAR,
    corrbond_syn_CDAR,
) = portfolio_syn(weight_syn_CDAR)

stackplts(weight_syn_CDAR)

t_syn_CDAR = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR, aum_syn_CDAR, returns_syn_CDAR, to_syn_CDAR, True
        ),
        performance_measures(
            weight_syn_CDAR, aum_syn_CDAR_tc, returns_syn_CDAR_tc, to_syn_CDAR, True
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR.columns = ["CDAR synth. - No costs", "CDAR synth. - With costs"]


# Optimal CDAR:

weight_syn_CDAR_risk = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CDAR risk optimization"):
    weight_syn_CDAR_risk[i] = pd.DataFrame(
        data=pf.CDAR_risk(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CDAR_risk,
    returns_syn_CDAR_risk,
    to_syn_CDAR_risk,
    aum_syn_CDAR_risk_tc,
    returns_syn_CDAR_risk_tc,
    corrsp_syn_CDAR_risk,
    corrbond_syn_CDAR_risk,
) = portfolio_syn(weight_syn_CDAR_risk)

stackplts(weight_syn_CDAR_risk)

t_syn_CDAR_risk = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_risk,
            aum_syn_CDAR_risk,
            returns_syn_CDAR_risk,
            to_syn_CDAR_risk,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_risk,
            aum_syn_CDAR_risk_tc,
            returns_syn_CDAR_risk_tc,
            to_syn_CDAR_risk,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_risk.columns = [
    "CDAR risk synth. - No costs",
    "CDAR risk synth. - With costs",
]


###############################################################################

# Minimum risk Omega:

weight_syn_Omegamin = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn Omegamin optimization"):
    weight_syn_Omegamin[i] = pd.DataFrame(
        data=pf.Omega_min(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_Omegamin,
    returns_syn_Omegamin,
    to_syn_Omegamin,
    aum_syn_Omegamin_tc,
    returns_syn_Omegamin_tc,
    corrsp_syn_Omegamin,
    corrbond_syn_Omegamin,
) = portfolio_syn(weight_syn_Omegamin)

stackplts(weight_syn_Omegamin)

t_syn_Omegamin = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamin,
            aum_syn_Omegamin,
            returns_syn_Omegamin,
            to_syn_Omegamin,
            True,
        ),
        performance_measures(
            weight_syn_Omegamin,
            aum_syn_Omegamin_tc,
            returns_syn_Omegamin_tc,
            to_syn_Omegamin,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamin.columns = [
    "Omega denum. synth - No costs",
    "Omega denum. synth - With costs",
]


# Optimal Omega:

weight_syn_Omegamax = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn Omegamax optimization"):
    weight_syn_Omegamax[i] = pd.DataFrame(
        data=pf.Omega_max(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_Omegamax,
    returns_syn_Omegamax,
    to_syn_Omegamax,
    aum_syn_Omegamax_tc,
    returns_syn_Omegamax_tc,
    corrsp_syn_Omegamax,
    corrbond_syn_Omegamax,
) = portfolio_syn(weight_syn_Omegamax)

stackplts(weight_syn_Omegamax)

t_syn_Omegamax = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamax,
            aum_syn_Omegamax,
            returns_syn_Omegamax,
            to_syn_Omegamax,
            True,
        ),
        performance_measures(
            weight_syn_Omegamax,
            aum_syn_Omegamax_tc,
            returns_syn_Omegamax_tc,
            to_syn_Omegamax,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamax.columns = [
    "Omega ratio synth.- No costs",
    "Omega ratio synth.- With costs",
]

###############################################################################

# Performance table

t_syn_P1 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_syn_CVAR,
        t_syn_CVAR_risk,
        t_syn_CDAR,
        t_syn_CDAR_risk,
        t_syn_Omegamin,
        t_syn_Omegamax,
    ],
    axis=1,
    ignore_index=False,
)
t_syn_P1.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_syn_P1.to_excel(writer, sheet_name="P1 - Synthetic")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using synthetic returns with costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_syn_CVAR_tc)
sns.lineplot(data=aum_syn_CVAR_risk_tc)
sns.lineplot(data=aum_syn_CDAR_tc)
sns.lineplot(data=aum_syn_CDAR_risk_tc)
sns.lineplot(data=aum_syn_Omegamin_tc)
sns.lineplot(data=aum_syn_Omegamax_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P1.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using historical returns without costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_syn_CVAR)
sns.lineplot(data=aum_syn_CVAR_risk)
sns.lineplot(data=aum_syn_CDAR)
sns.lineplot(data=aum_syn_CDAR_risk)
sns.lineplot(data=aum_syn_Omegamin)
sns.lineplot(data=aum_syn_Omegamax)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P1.png")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#         Part 2: Cost minimization added in objective function               #


###############################################################################
###############################################################################
###############################################################################
###############################################################################
# For some reason this variable needs to be redifined after some time:
rw_number = len(Returns) - rw
###############################################################################


# Minimum risk CVAR:

weight_CVAR_mint = [0] * rw_number
weight_CVAR_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])


for i in tqdm(range(rw_number), desc="CVAR mint"):
    weight_CVAR_mint[i] = pd.DataFrame(
        data=pf.CVAR(rw_returns[i], np.array(weight_CVAR_mint[i - 1]), 0.0005),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CVAR_mint,
    returns_CVAR_mint,
    to_CVAR_mint,
    aum_CVAR_mint_tc,
    returns_CVAR_mint_tc,
    corrsp_CVAR_mint,
    corrbond_CVAR_mint,
) = portfolio(weight_CVAR_mint)

turnover_plot(to_CVAR_mint, to_CVAR)

stackplts(weight_CVAR_mint)

t_CVAR_mint = pd.concat(
    [
        performance_measures(
            weight_CVAR_mint, aum_CVAR_mint, returns_CVAR_mint, to_CVAR_mint, False
        ),
        performance_measures(
            weight_CVAR_mint,
            aum_CVAR_mint_tc,
            returns_CVAR_mint_tc,
            to_CVAR_mint,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_mint.columns = ["Min t - CVAR - No costs", "Min t - CVAR - With costs"]


# Optimal CVAR:

weight_CVAR_risk_mint = [0] * rw_number
weight_CVAR_risk_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_number), desc="CVAR risk mint optimization"):
    weight_CVAR_risk_mint[i] = pd.DataFrame(
        data=pf.CVAR_risk(
            rw_returns[i], np.array(weight_CVAR_risk_mint[i - 1]), turnover_penalty
        ),  # 0.07
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_CVAR_risk_mint[i - 1],
            index=weight_CVAR_risk_mint[i].index,
            columns=weight_CVAR_risk_mint[i].columns,
        )
    else:
        w0 = weight_CVAR_risk_mint[i - 1]

    length_diff = len(weight_CVAR_risk_mint[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_CVAR_risk_mint[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2

del i


(
    aum_CVAR_risk_mint,
    returns_CVAR_risk_mint,
    to_CVAR_risk_mint,
    aum_CVAR_risk_mint_tc,
    returns_CVAR_risk_mint_tc,
    corrsp_CVAR_risk_mint,
    corrbond_CVAR_risk_mint,
) = portfolio(weight_CVAR_risk_mint)

turnover_plot(to_CVAR_risk_mint, to_CVAR_risk)

stackplts(weight_CVAR_risk_mint)

t_CVAR_risk_mint = pd.concat(
    [
        performance_measures(
            weight_CVAR_risk_mint,
            aum_CVAR_risk_mint,
            returns_CVAR_risk_mint,
            to_CVAR_risk_mint,
            False,
        ),
        performance_measures(
            weight_CVAR_risk_mint,
            aum_CVAR_risk_mint_tc,
            returns_CVAR_risk_mint_tc,
            to_CVAR_risk_mint,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_risk_mint.columns = ["Min t - CVAR - No costs", "Min t - CVAR - With costs"]


###############################################################################


# Minimum risk CDAR:

weight_CDAR_mint = [0] * rw_number
weight_CDAR_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_number), desc="CDAR mint optimization"):
    weight_CDAR_mint[i] = pd.DataFrame(
        data=pf.CDAR(rw_returns[i], np.array(weight_CDAR_mint[i - 1]), 0.001),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CDAR_mint,
    returns_CDAR_mint,
    to_CDAR_mint,
    aum_CDAR_mint_tc,
    returns_CDAR_mint_tc,
    corrsp_CDAR_mint,
    corrbond_CDAR_mint,
) = portfolio(weight_CDAR_mint)

turnover_plot(to_CDAR_mint, to_CDAR)

stackplts(weight_CDAR_mint)

t_CDAR_mint = pd.concat(
    [
        performance_measures(
            weight_CDAR_mint, aum_CDAR_mint, returns_CDAR_mint, to_CDAR_mint, False
        ),
        performance_measures(
            weight_CDAR_mint,
            aum_CDAR_mint_tc,
            returns_CDAR_mint_tc,
            to_CDAR_mint,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_mint.columns = ["Min t - CDAR - No costs", "Min t - CDAR - With costs"]


# Optimal CDAR:

weight_CDAR_risk_mint = [0] * rw_number
weight_CDAR_risk_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_number), desc="CDAR risk mint optimization"):
    weight_CDAR_risk_mint[i] = pd.DataFrame(
        data=pf.CDAR_risk(
            rw_returns[i], np.array(weight_CDAR_risk_mint[i - 1]), turnover_penalty
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_CDAR_risk_mint[i - 1],
            index=weight_CDAR_risk_mint[i].index,
            columns=weight_CDAR_risk_mint[i].columns,
        )
    else:
        w0 = weight_CDAR_risk_mint[i - 1]

    length_diff = len(weight_CDAR_risk_mint[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_CDAR_risk_mint[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2


del i


(
    aum_CDAR_risk_mint,
    returns_CDAR_risk_mint,
    to_CDAR_risk_mint,
    aum_CDAR_risk_mint_tc,
    returns_CDAR_risk_mint_tc,
    corrsp_CDAR_risk_mint,
    corrbond_CDAR_risk_mint,
) = portfolio(weight_CDAR_risk_mint)

turnover_plot(to_CDAR_risk_mint, to_CDAR_risk)

stackplts(weight_CDAR_risk_mint)

t_CDAR_risk_mint = pd.concat(
    [
        performance_measures(
            weight_CDAR_risk_mint,
            aum_CDAR_risk_mint,
            returns_CDAR_risk_mint,
            to_CDAR_risk_mint,
            False,
        ),
        performance_measures(
            weight_CDAR_risk_mint,
            aum_CDAR_risk_mint_tc,
            returns_CDAR_risk_mint_tc,
            to_CDAR_risk_mint,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_risk_mint.columns = ["Min t - CDAR - No costs", "Min t - CDAR - With costs"]


###############################################################################


# Minimum risk Omega:

weight_Omegamin_mint = [0] * rw_number
weight_Omegamin_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_number), desc="Omegamin mint optimization"):
    weight_Omegamin_mint[i] = pd.DataFrame(
        data=pf.Omega_min(rw_returns[i], np.array(weight_Omegamin_mint[i - 1]), 0.0002),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_Omegamin_mint,
    returns_Omegamin_mint,
    to_Omegamin_mint,
    aum_Omegamin_mint_tc,
    returns_Omegamin_mint_tc,
    corrsp_Omegamin_mint,
    corrbond_Omegamin_mint,
) = portfolio(weight_Omegamin_mint)


turnover_plot(to_Omegamin_mint, to_Omegamin)

stackplts(weight_Omegamin_mint)

t_Omegamin_mint = pd.concat(
    [
        performance_measures(
            weight_Omegamin_mint,
            aum_Omegamin_mint,
            returns_Omegamin_mint,
            to_Omegamin_mint,
            False,
        ),
        performance_measures(
            weight_Omegamin_mint,
            aum_Omegamin_mint_tc,
            returns_Omegamin_mint_tc,
            to_Omegamin_mint,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamin_mint.columns = [
    "Min t - Omega denum. - No costs",
    "Min t - Omega denum. - With costs",
]


# Optimal Omega:

weight_Omegamax_mint = [0] * rw_number
weight_Omegamax_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_number), desc="Omegamax mint optimization"):
    weight_Omegamax_mint[i] = pd.DataFrame(
        data=pf.Omega_max(
            rw_returns[i], np.array(weight_Omegamax_mint[i - 1]), turnover_penalty
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_Omegamax_mint[i - 1],
            index=weight_Omegamax_mint[i].index,
            columns=weight_Omegamax_mint[i].columns,
        )
    else:
        w0 = weight_Omegamax_mint[i - 1]

    length_diff = len(weight_Omegamax_mint[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_Omegamax_mint[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2


del i


(
    aum_Omegamax_mint,
    returns_Omegamax_mint,
    to_Omegamax_mint,
    aum_Omegamax_mint_tc,
    returns_Omegamax_mint_tc,
    corrsp_Omegamax_mint,
    corrbond_Omegamax_mint,
) = portfolio(weight_Omegamax_mint)

turnover_plot(to_Omegamax_mint, to_Omegamax)

stackplts(weight_Omegamax_mint)

t_Omegamax_mint = pd.concat(
    [
        performance_measures(
            weight_Omegamax_mint,
            aum_Omegamax_mint,
            returns_Omegamax_mint,
            to_Omegamax_mint,
            False,
        ),
        performance_measures(
            weight_Omegamax_mint,
            aum_Omegamax_mint_tc,
            returns_Omegamax_mint_tc,
            to_Omegamax_mint,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamax_mint.columns = [
    "Min t - Omega ratio - No costs",
    "Min t - Omega ratio - With costs",
]

###############################################################################

# Performance table

t_historical_P2 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_CVAR_mint,
        t_CVAR_risk_mint,
        t_CDAR_mint,
        t_CDAR_risk_mint,
        t_Omegamin_mint,
        t_Omegamax_mint,
    ],
    axis=1,
    ignore_index=False,
)

t_historical_P2.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_historical_P2.to_excel(writer, sheet_name="P2 - Historical")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using historical returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_CVAR_mint_tc)
sns.lineplot(data=aum_CVAR_risk_mint_tc)
sns.lineplot(data=aum_CDAR_mint_tc)
sns.lineplot(data=aum_CDAR_risk_mint_tc)
sns.lineplot(data=aum_Omegamin_mint_tc)
sns.lineplot(data=aum_Omegamax_mint_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P2.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using historical returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_CVAR_mint)
sns.lineplot(data=aum_CVAR_risk_mint)
sns.lineplot(data=aum_CDAR_mint)
sns.lineplot(data=aum_CDAR_risk_mint)
sns.lineplot(data=aum_Omegamin_mint)
sns.lineplot(data=aum_Omegamax_mint)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P2.png")


###############################################################################


#                    Optimization with synthetic returns                      #


###############################################################################


# Minimum risk cvar

weight_syn_CVAR_mint = [0] * rw_corr_number
weight_syn_CVAR_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_corr_number), desc="syn CVAR mint optimization"):
    weight_syn_CVAR_mint[i] = pd.DataFrame(
        data=pf.CVAR(rw_corr_returns[i], np.array(weight_syn_CVAR_mint[i - 1]), 0.0005),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CVAR_mint,
    returns_syn_CVAR_mint,
    to_syn_CVAR_mint,
    aum_syn_CVAR_mint_tc,
    returns_syn_CVAR_mint_tc,
    corrsp_syn_CVAR_mint,
    corrbond_syn_CVAR_mint,
) = portfolio_syn(weight_syn_CVAR_mint)

turnover_plot(to_syn_CVAR_mint, to_syn_CVAR)

stackplts(weight_syn_CVAR_mint)

t_syn_CVAR_mint = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_mint,
            aum_syn_CVAR_mint,
            returns_syn_CVAR_mint,
            to_syn_CVAR_mint,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_mint,
            aum_syn_CVAR_mint_tc,
            returns_syn_CVAR_mint_tc,
            to_syn_CVAR_mint,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_mint.columns = [
    "Min t - CVAR synth. - No costs",
    "Min t - CVAR synth. - With costs",
]


# Optimal CVAR:

weight_syn_CVAR_risk_mint = [0] * rw_corr_number
weight_syn_CVAR_risk_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])


turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_corr_number), desc="syn CVAR risk mint optimization"):

    weight_syn_CVAR_risk_mint[i] = pd.DataFrame(
        data=pf.CVAR_risk(
            rw_corr_returns[i],
            np.array(weight_syn_CVAR_risk_mint[i - 1]),
            turnover_penalty,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_syn_CVAR_risk_mint[i - 1],
            index=weight_syn_CVAR_risk_mint[i].index,
            columns=weight_syn_CVAR_risk_mint[i].columns,
        )
    else:
        w0 = weight_syn_CVAR_risk_mint[i - 1]

    length_diff = len(weight_syn_CVAR_risk_mint[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_syn_CVAR_risk_mint[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2

del i


(
    aum_syn_CVAR_risk_mint,
    returns_syn_CVAR_risk_mint,
    to_syn_CVAR_risk_mint,
    aum_syn_CVAR_risk_mint_tc,
    returns_syn_CVAR_risk_mint_tc,
    corrsp_syn_CVAR_risk_mint,
    corrbond_syn_CVAR_risk_mint,
) = portfolio_syn(weight_syn_CVAR_risk_mint)

turnover_plot(to_syn_CVAR_risk_mint, to_syn_CVAR_risk)

stackplts(weight_syn_CVAR_risk_mint)

t_syn_CVAR_risk_mint = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_risk_mint,
            aum_syn_CVAR_risk_mint,
            returns_syn_CVAR_risk_mint,
            to_syn_CVAR_risk_mint,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_risk_mint,
            aum_syn_CVAR_risk_mint_tc,
            returns_syn_CVAR_risk_mint_tc,
            to_syn_CVAR_risk_mint,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_risk_mint.columns = [
    "Min t - CVAR synth. - No costs",
    "Min t - CVAR synth. - With costs",
]


###############################################################################


# Minimum risk CDAR:

weight_syn_CDAR_mint = [0] * rw_corr_number
weight_syn_CDAR_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_corr_number), desc="syn CDAR mint optimization"):
    weight_syn_CDAR_mint[i] = pd.DataFrame(
        data=pf.CDAR(rw_corr_returns[i], np.array(weight_syn_CDAR_mint[i - 1]), 0.001),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )


del i


(
    aum_syn_CDAR_mint,
    returns_syn_CDAR_mint,
    to_syn_CDAR_mint,
    aum_syn_CDAR_mint_tc,
    returns_syn_CDAR_mint_tc,
    corrsp_syn_CDAR_mint,
    corrbond_syn_CDAR_mint,
) = portfolio_syn(weight_syn_CDAR_mint)

turnover_plot(to_syn_CDAR_mint, to_syn_CDAR)

stackplts(weight_syn_CDAR_mint)

t_syn_CDAR_mint = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_mint,
            aum_syn_CDAR_mint,
            returns_syn_CDAR_mint,
            to_syn_CDAR_mint,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_mint,
            aum_syn_CDAR_mint_tc,
            returns_syn_CDAR_mint_tc,
            to_syn_CDAR_mint,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_mint.columns = [
    "Min t - CDAR synth. - No costs",
    "Min t - CDAR synth. - With costs",
]


# Optimal CDAR:

weight_syn_CDAR_risk_mint = [0] * rw_corr_number
weight_syn_CDAR_risk_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])


turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_corr_number), desc="syn CDAR risk mint optimization"):
    weight_syn_CDAR_risk_mint[i] = pd.DataFrame(
        data=pf.CDAR_risk(
            rw_corr_returns[i],
            np.array(weight_syn_CDAR_risk_mint[i - 1]),
            turnover_penalty,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_syn_CDAR_risk_mint[i - 1],
            index=weight_syn_CDAR_risk_mint[i].index,
            columns=weight_syn_CDAR_risk_mint[i].columns,
        )
    else:
        w0 = weight_syn_CDAR_risk_mint[i - 1]

    length_diff = len(weight_syn_CDAR_risk_mint[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_syn_CDAR_risk_mint[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2


del i


(
    aum_syn_CDAR_risk_mint,
    returns_syn_CDAR_risk_mint,
    to_syn_CDAR_risk_mint,
    aum_syn_CDAR_risk_mint_tc,
    returns_syn_CDAR_risk_mint_tc,
    corrsp_syn_CDAR_risk_mint,
    corrbond_syn_CDAR_risk_mint,
) = portfolio_syn(weight_syn_CDAR_risk_mint)

turnover_plot(to_syn_CDAR_risk_mint, to_syn_CDAR_risk)

stackplts(weight_syn_CDAR_risk_mint)

t_syn_CDAR_risk_mint = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_risk_mint,
            aum_syn_CDAR_risk_mint,
            returns_syn_CDAR_risk_mint,
            to_syn_CDAR_risk_mint,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_risk_mint,
            aum_syn_CDAR_risk_mint_tc,
            returns_syn_CDAR_risk_mint_tc,
            to_syn_CDAR_risk_mint,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_risk_mint.columns = [
    "Min t - CDAR synth. - No costs",
    "Min t - CDAR synth. - With costs",
]


###############################################################################


# Minimum risk Omega:

weight_syn_Omegamin_mint = [0] * rw_corr_number
weight_syn_Omegamin_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_corr_number), desc="syn Omegamin mint optimization"):
    weight_syn_Omegamin_mint[i] = pd.DataFrame(
        data=pf.Omega_min(
            rw_corr_returns[i], np.array(weight_syn_Omegamin_mint[i - 1]), 0.0002
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_Omegamin_mint,
    returns_syn_Omegamin_mint,
    to_syn_Omegamin_mint,
    aum_syn_Omegamin_mint_tc,
    returns_syn_Omegamin_mint_tc,
    corrsp_syn_Omegamin_mint,
    corrbond_syn_Omegamin_mint,
) = portfolio_syn(weight_syn_Omegamin_mint)


turnover_plot(to_syn_Omegamin_mint, to_syn_Omegamin)

stackplts(weight_syn_Omegamin_mint)


t_syn_Omegamin_mint = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamin_mint,
            aum_syn_Omegamin_mint,
            returns_syn_Omegamin_mint,
            to_syn_Omegamin_mint,
            True,
        ),
        performance_measures(
            weight_syn_Omegamin_mint,
            aum_syn_Omegamin_mint_tc,
            returns_syn_Omegamin_mint_tc,
            to_syn_Omegamin_mint,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamin_mint.columns = [
    "Min t - Omega denum. synth - No costs",
    "Min t - Omega denum. synth - With costs",
]


# Optimal Omega:

weight_syn_Omegamax_mint = [0] * rw_corr_number
weight_syn_Omegamax_mint[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])


turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_corr_number), desc="syn Omegamax mint optimization"):
    weight_syn_Omegamax_mint[i] = pd.DataFrame(
        data=pf.Omega_max(
            rw_corr_returns[i],
            np.array(weight_syn_Omegamax_mint[i - 1]),
            turnover_penalty,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_syn_Omegamax_mint[i - 1],
            index=weight_syn_Omegamax_mint[i].index,
            columns=weight_syn_Omegamax_mint[i].columns,
        )
    else:
        w0 = weight_syn_Omegamax_mint[i - 1]

    length_diff = len(weight_syn_Omegamax_mint[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_syn_Omegamax_mint[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2

del i


(
    aum_syn_Omegamax_mint,
    returns_syn_Omegamax_mint,
    to_syn_Omegamax_mint,
    aum_syn_Omegamax_mint_tc,
    returns_syn_Omegamax_mint_tc,
    corrsp_syn_Omegamax_mint,
    corrbond_syn_Omegamax_mint,
) = portfolio_syn(weight_syn_Omegamax_mint)

turnover_plot(to_syn_Omegamax_mint, to_syn_Omegamax)

stackplts(weight_syn_Omegamax_mint)

t_syn_Omegamax_mint = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamax_mint,
            aum_syn_Omegamax_mint,
            returns_syn_Omegamax_mint,
            to_syn_Omegamax_mint,
            True,
        ),
        performance_measures(
            weight_syn_Omegamax_mint,
            aum_syn_Omegamax_mint_tc,
            returns_syn_Omegamax_mint_tc,
            to_syn_Omegamax_mint,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamax_mint.columns = [
    "Min t - Omega ratio synth.- No costs",
    "Min t - Omega ratio synth.- With costs",
]


###############################################################################

# Performance table

t_syn_P2 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_syn_CVAR_mint,
        t_syn_CVAR_risk_mint,
        t_syn_CDAR_mint,
        t_syn_CDAR_risk_mint,
        t_syn_Omegamin_mint,
        t_syn_Omegamax_mint,
    ],
    axis=1,
    ignore_index=False,
)

t_syn_P2.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_syn_P2.to_excel(writer, sheet_name="P2 - Synthetic")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using synthetic returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_syn_CVAR_mint_tc)
sns.lineplot(data=aum_syn_CVAR_risk_mint_tc)
sns.lineplot(data=aum_syn_CDAR_mint_tc)
sns.lineplot(data=aum_syn_CDAR_risk_mint_tc)
sns.lineplot(data=aum_syn_Omegamin_mint_tc)
sns.lineplot(data=aum_syn_Omegamax_mint_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P2.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using synthetic returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_syn_CVAR_mint)
sns.lineplot(data=aum_syn_CVAR_risk_mint)
sns.lineplot(data=aum_syn_CDAR_mint)
sns.lineplot(data=aum_syn_CDAR_risk_mint)
sns.lineplot(data=aum_syn_Omegamin_mint)
sns.lineplot(data=aum_syn_Omegamax_mint)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P2.png")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#     Part 3: Portfolio correlation constraints on stocks and bonds added     #


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# Constraint parameter on maximium correlation of the portfolio to asset class
# FIXME: Parameters of the constraints
MaxCorrelationSP = 0.35
MaxCorrelationBond = 0.25

# Minimum risk CVAR:

weight_CVAR_cons = [0] * rw_number

for i in tqdm(range(rw_number), desc="CVAR optimization under constraint"):
    weight_CVAR_cons[i] = pd.DataFrame(
        data=pf.CVAR(
            rw_returns[i],
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CVAR_cons,
    returns_CVAR_cons,
    to_CVAR_cons,
    aum_CVAR_cons_tc,
    returns_CVAR_cons_tc,
    corrsp_CVAR_cons,
    corrbond_CVAR_cons,
) = portfolio(weight_CVAR_cons)

correlation_plot(
    corrsp_CVAR_cons,
    corrbond_CVAR_cons,
    corrsp_CVAR,
    corrbond_CVAR,
)

stackplts(weight_CVAR_cons)

t_CVAR_cons = pd.concat(
    [
        performance_measures(
            weight_CVAR_cons, aum_CVAR_cons, returns_CVAR_cons, to_CVAR_cons, False
        ),
        performance_measures(
            weight_CVAR_cons,
            aum_CVAR_cons_tc,
            returns_CVAR_cons_tc,
            to_CVAR_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_cons.columns = [
    "Corr - CVAR - No costs",
    "Corr - CVAR - With costs",
]


# Optimal CVAR:

weight_CVAR_risk_cons = [0] * rw_number

for i in tqdm(range(rw_number), desc="CVAR risk optimization under constraint"):
    weight_CVAR_risk_cons[i] = pd.DataFrame(
        data=pf.CVAR_risk(
            rw_returns[i],
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CVAR_risk_cons,
    returns_CVAR_risk_cons,
    to_CVAR_risk_cons,
    aum_CVAR_risk_cons_tc,
    returns_CVAR_risk_cons_tc,
    corrsp_CVAR_risk_cons,
    corrbond_CVAR_risk_cons,
) = portfolio(weight_CVAR_risk_cons)

correlation_plot(
    corrsp_CVAR_risk_cons,
    corrbond_CVAR_risk_cons,
    corrsp_CVAR_risk,
    corrbond_CVAR_risk,
)

stackplts(weight_CVAR_risk_cons)

t_CVAR_risk_cons = pd.concat(
    [
        performance_measures(
            weight_CVAR_risk_cons,
            aum_CVAR_risk_cons,
            returns_CVAR_risk_cons,
            to_CVAR_risk_cons,
            False,
        ),
        performance_measures(
            weight_CVAR_risk_cons,
            aum_CVAR_risk_cons_tc,
            returns_CVAR_risk_cons_tc,
            to_CVAR_risk_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_risk_cons.columns = [
    "Corr - CVAR Risk - No costs",
    "Corr - CVAR Risk - With costs",
]

###############################################################################


# Minimum risk CDAR:

weight_CDAR_cons = [0] * rw_number

for i in tqdm(range(rw_number), desc="CDAR optimization under constraint"):
    weight_CDAR_cons[i] = pd.DataFrame(
        data=pf.CDAR(
            rw_returns[i],
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CDAR_cons,
    returns_CDAR_cons,
    to_CDAR_cons,
    aum_CDAR_cons_tc,
    returns_CDAR_cons_tc,
    corrsp_CDAR_cons,
    corrbond_CDAR_cons,
) = portfolio(weight_CDAR_cons)

correlation_plot(
    corrsp_CDAR_cons,
    corrbond_CDAR_cons,
    corrsp_CDAR,
    corrbond_CDAR,
)

stackplts(weight_CDAR_cons)

t_CDAR_cons = pd.concat(
    [
        performance_measures(
            weight_CDAR_cons, aum_CDAR_cons, returns_CDAR_cons, to_CDAR_cons, False
        ),
        performance_measures(
            weight_CDAR_cons,
            aum_CDAR_cons_tc,
            returns_CDAR_cons_tc,
            to_CDAR_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_cons.columns = [
    "Corr - CDAR - No costs",
    "Corr - CDAR - With costs",
]


# Optimal CDAR:

weight_CDAR_risk_cons = [0] * rw_number

for i in tqdm(range(rw_number), desc="CDAR risk optimization under constraint"):
    weight_CDAR_risk_cons[i] = pd.DataFrame(
        data=pf.CDAR_risk(
            rw_returns[i],
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CDAR_risk_cons,
    returns_CDAR_risk_cons,
    to_CDAR_risk_cons,
    aum_CDAR_risk_cons_tc,
    returns_CDAR_risk_cons_tc,
    corrsp_CDAR_risk_cons,
    corrbond_CDAR_risk_cons,
) = portfolio(weight_CDAR_risk_cons)

correlation_plot(
    corrsp_CDAR_risk_cons,
    corrbond_CDAR_risk_cons,
    corrsp_CDAR_risk,
    corrbond_CDAR_risk,
)

stackplts(weight_CDAR_risk_cons)

t_CDAR_risk_cons = pd.concat(
    [
        performance_measures(
            weight_CDAR_risk_cons,
            aum_CDAR_risk_cons,
            returns_CDAR_risk_cons,
            to_CDAR_risk_cons,
            False,
        ),
        performance_measures(
            weight_CDAR_risk_cons,
            aum_CDAR_risk_cons_tc,
            returns_CDAR_risk_cons_tc,
            to_CDAR_risk_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_risk_cons.columns = [
    "Corr - CDAR Risk - No costs",
    "Corr - CDAR Risk - With costs",
]


###############################################################################


# Minimum risk Omega:

weight_Omegamin_cons = [0] * rw_number

for i in tqdm(range(rw_number), desc="Omegamin optimization under constraint"):
    weight_Omegamin_cons[i] = pd.DataFrame(
        data=pf.Omega_min(
            rw_returns[i],
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_Omegamin_cons,
    returns_Omegamin_cons,
    to_Omegamin_cons,
    aum_Omegamin_cons_tc,
    returns_Omegamin_cons_tc,
    corrsp_Omegamin_cons,
    corrbond_Omegamin_cons,
) = portfolio(weight_Omegamin_cons)

correlation_plot(
    corrsp_Omegamin_cons,
    corrbond_Omegamin_cons,
    corrsp_Omegamin,
    corrbond_Omegamin,
)

stackplts(weight_Omegamin_cons)

t_Omegamin_cons = pd.concat(
    [
        performance_measures(
            weight_Omegamin_cons,
            aum_Omegamin_cons,
            returns_Omegamin_cons,
            to_Omegamin_cons,
            False,
        ),
        performance_measures(
            weight_Omegamin_cons,
            aum_Omegamin_cons_tc,
            returns_Omegamin_cons_tc,
            to_Omegamin_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamin_cons.columns = [
    "Corr - Omegamin - No costs",
    "Corr - Omegamin - With costs",
]


# Optimal Omega:

weight_Omegamax_cons = [0] * rw_number

for i in tqdm(range(rw_number), desc="Omegamax optimization under constraint"):
    weight_Omegamax_cons[i] = pd.DataFrame(
        data=pf.Omega_max(
            rw_returns[i],
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_Omegamax_cons,
    returns_Omegamax_cons,
    to_Omegamax_cons,
    aum_Omegamax_cons_tc,
    returns_Omegamax_cons_tc,
    corrsp_Omegamax_cons,
    corrbond_Omegamax_cons,
) = portfolio(weight_Omegamax_cons)

correlation_plot(
    corrsp_Omegamax_cons,
    corrbond_Omegamax_cons,
    corrsp_Omegamax,
    corrbond_Omegamax,
)

stackplts(weight_Omegamax_cons)

t_Omegamax_cons = pd.concat(
    [
        performance_measures(
            weight_Omegamax_cons,
            aum_Omegamax_cons,
            returns_Omegamax_cons,
            to_Omegamax_cons,
            False,
        ),
        performance_measures(
            weight_Omegamax_cons,
            aum_Omegamax_cons_tc,
            returns_Omegamax_cons_tc,
            to_Omegamax_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamax_cons.columns = [
    "Corr - Omegamax - No costs",
    "Corr - Omegamax - With costs",
]

###############################################################################

# Performance table

t_historical_P3 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_CVAR_cons,
        t_CVAR_risk_cons,
        t_CDAR_cons,
        t_CDAR_risk_cons,
        t_Omegamin_cons,
        t_Omegamax_cons,
    ],
    axis=1,
    ignore_index=False,
)

t_historical_P3.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_historical_P3.to_excel(writer, sheet_name="P3 - Historical")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with correlation constraints computed using historical returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_CVAR_cons_tc)
sns.lineplot(data=aum_CVAR_risk_cons_tc)
sns.lineplot(data=aum_CDAR_cons_tc)
sns.lineplot(data=aum_CDAR_risk_cons_tc)
sns.lineplot(data=aum_Omegamin_cons_tc)
sns.lineplot(data=aum_Omegamax_cons_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P3.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with correlation constraints computed using historical returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_CVAR_cons)
sns.lineplot(data=aum_CVAR_risk_cons)
sns.lineplot(data=aum_CDAR_cons)
sns.lineplot(data=aum_CDAR_risk_cons)
sns.lineplot(data=aum_Omegamin_cons)
sns.lineplot(data=aum_Omegamax_cons)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P3.png")


###############################################################################


#                    Optimization with synthetic returns and constraint on portfolio correlations             (syn + cons)      #


###############################################################################


# Minimum risk CVAR

weight_syn_CVAR_cons = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CVAR optimization under constraint"):
    weight_syn_CVAR_cons[i] = pd.DataFrame(
        data=pf.CVAR(
            rw_corr_returns[i],
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CVAR_cons,
    returns_syn_CVAR_cons,
    to_syn_CVAR_cons,
    aum_syn_CVAR_cons_tc,
    returns_syn_CVAR_cons_tc,
    corrsp_syn_CVAR_cons,
    corrbond_syn_CVAR_cons,
) = portfolio_syn(weight_syn_CVAR_cons)

correlation_plot(
    corrsp_syn_CVAR_cons,
    corrbond_syn_CVAR_cons,
    corrsp_syn_CVAR,
    corrbond_syn_CVAR,
)

stackplts(weight_syn_CVAR_cons)

t_syn_CVAR_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_cons,
            aum_syn_CVAR_cons,
            returns_syn_CVAR_cons,
            to_syn_CVAR_cons,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_cons,
            aum_syn_CVAR_cons_tc,
            returns_syn_CVAR_cons_tc,
            to_syn_CVAR_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_cons.columns = [
    "Corr - CVAR synth. - No costs",
    "Corr - CVAR synth. - With costs",
]


# Optimal CVAR:

weight_syn_CVAR_risk_cons = [0] * rw_corr_number

for i in tqdm(
    range(rw_corr_number), desc="syn CVAR risk optimization under constraint"
):
    weight_syn_CVAR_risk_cons[i] = pd.DataFrame(
        data=pf.CVAR_risk(
            rw_corr_returns[i],
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CVAR_risk_cons,
    returns_syn_CVAR_risk_cons,
    to_syn_CVAR_risk_cons,
    aum_syn_CVAR_risk_cons_tc,
    returns_syn_CVAR_risk_cons_tc,
    corrsp_syn_CVAR_risk_cons,
    corrbond_syn_CVAR_risk_cons,
) = portfolio_syn(weight_syn_CVAR_risk_cons)


correlation_plot(
    corrsp_syn_CVAR_risk_cons,
    corrbond_syn_CVAR_risk_cons,
    corrsp_syn_CVAR_risk,
    corrbond_syn_CVAR_risk,
)


stackplts(weight_syn_CVAR_risk_cons)

t_syn_CVAR_risk_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_risk_cons,
            aum_syn_CVAR_risk_cons,
            returns_syn_CVAR_risk_cons,
            to_syn_CVAR_risk_cons,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_risk_cons,
            aum_syn_CVAR_risk_cons_tc,
            returns_syn_CVAR_risk_cons_tc,
            to_syn_CVAR_risk_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_risk_cons.columns = [
    "Corr - CVAR risk synth. - No costs",
    "Corr - CVAR risk synth. - With costs",
]


###############################################################################


# Minimum risk CDAR:

weight_syn_CDAR_cons = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CDAR optimization under constraint"):
    weight_syn_CDAR_cons[i] = pd.DataFrame(
        data=pf.CDAR(
            rw_corr_returns[i],
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CDAR_cons,
    returns_syn_CDAR_cons,
    to_syn_CDAR_cons,
    aum_syn_CDAR_cons_tc,
    returns_syn_CDAR_cons_tc,
    corrsp_syn_CDAR_cons,
    corrbond_syn_CDAR_cons,
) = portfolio_syn(weight_syn_CDAR_cons)

correlation_plot(
    corrsp_syn_CDAR_cons,
    corrbond_syn_CDAR_cons,
    corrsp_syn_CDAR,
    corrbond_syn_CDAR,
)

stackplts(weight_syn_CDAR_cons)

t_syn_CDAR_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_cons,
            aum_syn_CDAR_cons,
            returns_syn_CDAR_cons,
            to_syn_CDAR_cons,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_cons,
            aum_syn_CDAR_cons_tc,
            returns_syn_CDAR_cons_tc,
            to_syn_CDAR_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_cons.columns = [
    "Corr - CDAR synth. - No costs",
    "Corr - CDAR synth. - With costs",
]


# Optimal CDAR:

weight_syn_CDAR_risk_cons = [0] * rw_corr_number

for i in tqdm(
    range(rw_corr_number), desc="syn CDAR risk optimization under constraint"
):
    weight_syn_CDAR_risk_cons[i] = pd.DataFrame(
        data=pf.CDAR_risk(
            rw_corr_returns[i],
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CDAR_risk_cons,
    returns_syn_CDAR_risk_cons,
    to_syn_CDAR_risk_cons,
    aum_syn_CDAR_risk_cons_tc,
    returns_syn_CDAR_risk_cons_tc,
    corrsp_syn_CDAR_risk_cons,
    corrbond_syn_CDAR_risk_cons,
) = portfolio_syn(weight_syn_CDAR_risk_cons)


correlation_plot(
    corrsp_syn_CDAR_risk_cons,
    corrbond_syn_CDAR_risk_cons,
    corrsp_syn_CDAR_risk,
    corrbond_syn_CDAR_risk,
)

stackplts(weight_syn_CDAR_risk_cons)

t_syn_CDAR_risk_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_risk_cons,
            aum_syn_CDAR_risk_cons,
            returns_syn_CDAR_risk_cons,
            to_syn_CDAR_risk_cons,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_risk_cons,
            aum_syn_CDAR_risk_cons_tc,
            returns_syn_CDAR_risk_cons_tc,
            to_syn_CDAR_risk_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_risk_cons.columns = [
    "Corr - CDAR risk synth. - No costs",
    "Corr - CDAR risk synth. - With costs",
]


###############################################################################

# Minimum risk Omega:

weight_syn_Omegamin_cons = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn Omegamin optimization under constraint"):
    weight_syn_Omegamin_cons[i] = pd.DataFrame(
        data=pf.Omega_min(
            rw_corr_returns[i],
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_Omegamin_cons,
    returns_syn_Omegamin_cons,
    to_syn_Omegamin_cons,
    aum_syn_Omegamin_cons_tc,
    returns_syn_Omegamin_cons_tc,
    corrsp_syn_Omegamin_cons,
    corrbond_syn_Omegamin_cons,
) = portfolio_syn(weight_syn_Omegamin_cons)


correlation_plot(
    corrsp_syn_Omegamin_cons,
    corrbond_syn_Omegamin_cons,
    corrsp_syn_Omegamin,
    corrbond_syn_Omegamin,
)

stackplts(weight_syn_Omegamin_cons)

t_syn_Omegamin_cons = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamin_cons,
            aum_syn_Omegamin_cons,
            returns_syn_Omegamin_cons,
            to_syn_Omegamin_cons,
            True,
        ),
        performance_measures(
            weight_syn_Omegamin_cons,
            aum_syn_Omegamin_cons_tc,
            returns_syn_Omegamin_cons_tc,
            to_syn_Omegamin_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamin_cons.columns = [
    "Corr - Omegamin synth. - No costs",
    "Corr - Omegamin synth. - With costs",
]


# Optimal Omega:

weight_syn_Omegamax_cons = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn Omegamax optimization under constraint"):
    weight_syn_Omegamax_cons[i] = pd.DataFrame(
        data=pf.Omega_max(
            rw_corr_returns[i],
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_Omegamax_cons,
    returns_syn_Omegamax_cons,
    to_syn_Omegamax_cons,
    aum_syn_Omegamax_cons_tc,
    returns_syn_Omegamax_cons_tc,
    corrsp_syn_Omegamax_cons,
    corrbond_syn_Omegamax_cons,
) = portfolio_syn(weight_syn_Omegamax_cons)


correlation_plot(
    corrsp_syn_Omegamax_cons,
    corrbond_syn_Omegamax_cons,
    corrsp_syn_Omegamax,
    corrbond_syn_Omegamax,
)

stackplts(weight_syn_Omegamax_cons)

t_syn_Omegamax_cons = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamax_cons,
            aum_syn_Omegamax_cons,
            returns_syn_Omegamax_cons,
            to_syn_Omegamax_cons,
            True,
        ),
        performance_measures(
            weight_syn_Omegamax_cons,
            aum_syn_Omegamax_cons_tc,
            returns_syn_Omegamax_cons_tc,
            to_syn_Omegamax_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamax_cons.columns = [
    "Corr - Omegamax synth. - No costs",
    "Corr - Omegamax synth. - With costs",
]


###############################################################################

# Performance table

t_syn_P3 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_syn_CVAR_cons,
        t_syn_CVAR_risk_cons,
        t_syn_CDAR_cons,
        t_syn_CDAR_risk_cons,
        t_syn_Omegamin_cons,
        t_syn_Omegamax_cons,
    ],
    axis=1,
    ignore_index=False,
)

t_syn_P3.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_syn_P3.to_excel(writer, sheet_name="P3 - Synthetic")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with correlation constraints computed using synthetic returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_syn_CVAR_cons_tc)
sns.lineplot(data=aum_syn_CVAR_risk_cons_tc)
sns.lineplot(data=aum_syn_CDAR_cons_tc)
sns.lineplot(data=aum_syn_CDAR_risk_cons_tc)
sns.lineplot(data=aum_syn_Omegamin_cons_tc)
sns.lineplot(data=aum_syn_Omegamax_cons_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P3.png")


# Without costs:
sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using synthetic returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_syn_CVAR_cons)
sns.lineplot(data=aum_syn_CVAR_risk_cons)
sns.lineplot(data=aum_syn_CDAR_cons)
sns.lineplot(data=aum_syn_CDAR_risk_cons)
sns.lineplot(data=aum_syn_Omegamin_cons)
sns.lineplot(data=aum_syn_Omegamax_cons)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P3.png")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#          Part 4: Both cost minimization and correlation constraints         #


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# Minimum risk CVAR:

weight_CVAR_mint_cons = [0] * rw_number
weight_CVAR_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])


for i in tqdm(range(rw_number), desc="CVAR mint under constraint"):
    weight_CVAR_mint_cons[i] = pd.DataFrame(
        data=pf.CVAR(
            rw_returns[i],
            np.array(weight_CVAR_mint_cons[i - 1]),
            0.0005,
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CVAR_mint_cons,
    returns_CVAR_mint_cons,
    to_CVAR_mint_cons,
    aum_CVAR_mint_cons_tc,
    returns_CVAR_mint_cons_tc,
    corrsp_CVAR_mint_cons,
    corrbond_CVAR_mint_cons,
) = portfolio(weight_CVAR_mint_cons)

turnover_plot(to_CVAR_mint_cons, to_CVAR)


correlation_plot(
    corrsp_CVAR_mint_cons,
    corrbond_CVAR_mint_cons,
    corrsp_CVAR,
    corrbond_CVAR,
)

stackplts(weight_CVAR_mint_cons)

t_CVAR_mint_cons = pd.concat(
    [
        performance_measures(
            weight_CVAR_mint_cons,
            aum_CVAR_mint_cons,
            returns_CVAR_mint_cons,
            to_CVAR_mint_cons,
            False,
        ),
        performance_measures(
            weight_CVAR_mint_cons,
            aum_CVAR_mint_cons_tc,
            returns_CVAR_mint_cons_tc,
            to_CVAR_mint_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_mint_cons.columns = [
    "Corr & Mint - CVAR - No costs",
    "Corr & Mint - CVAR - With costs",
]


# Optimal CVAR:

weight_CVAR_risk_mint_cons = [0] * rw_number
weight_CVAR_risk_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_number), desc="CVAR risk mint optimization under constraint"):
    weight_CVAR_risk_mint_cons[i] = pd.DataFrame(
        data=pf.CVAR_risk(
            rw_returns[i],
            np.array(weight_CVAR_risk_mint_cons[i - 1]),
            turnover_penalty,
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_CVAR_risk_mint_cons[i - 1],
            index=weight_CVAR_risk_mint_cons[i].index,
            columns=weight_CVAR_risk_mint_cons[i].columns,
        )
    else:
        w0 = weight_CVAR_risk_mint_cons[i - 1]

    length_diff = len(weight_CVAR_risk_mint_cons[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_CVAR_risk_mint_cons[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2

del i


(
    aum_CVAR_risk_mint_cons,
    returns_CVAR_risk_mint_cons,
    to_CVAR_risk_mint_cons,
    aum_CVAR_risk_mint_cons_tc,
    returns_CVAR_risk_mint_cons_tc,
    corrsp_CVAR_risk_mint_cons,
    corrbond_CVAR_risk_mint_cons,
) = portfolio(weight_CVAR_risk_mint_cons)

turnover_plot(to_CVAR_risk_mint_cons, to_CVAR_risk)

correlation_plot(
    corrsp_CVAR_risk_mint_cons,
    corrbond_CVAR_risk_mint_cons,
    corrsp_CVAR_risk,
    corrbond_CVAR_risk,
)

stackplts(weight_CVAR_risk_mint_cons)

t_CVAR_risk_mint_cons = pd.concat(
    [
        performance_measures(
            weight_CVAR_risk_mint_cons,
            aum_CVAR_risk_mint_cons,
            returns_CVAR_risk_mint_cons,
            to_CVAR_risk_mint_cons,
            False,
        ),
        performance_measures(
            weight_CVAR_risk_mint_cons,
            aum_CVAR_risk_mint_cons_tc,
            returns_CVAR_risk_mint_cons_tc,
            to_CVAR_risk_mint_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CVAR_risk_mint_cons.columns = [
    "Corr & Mint - CVAR risk - No costs",
    "Corr & Mint - CVAR risk - With costs",
]


###############################################################################


# Minimum risk CDAR:

weight_CDAR_mint_cons = [0] * rw_number
weight_CDAR_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_number), desc="CDAR mint optimization under constraint"):
    weight_CDAR_mint_cons[i] = pd.DataFrame(
        data=pf.CDAR(
            rw_returns[i],
            np.array(weight_CDAR_mint_cons[i - 1]),
            0.001,
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_CDAR_mint_cons,
    returns_CDAR_mint_cons,
    to_CDAR_mint_cons,
    aum_CDAR_mint_cons_tc,
    returns_CDAR_mint_cons_tc,
    corrsp_CDAR_mint_cons,
    corrbond_CDAR_mint_cons,
) = portfolio(weight_CDAR_mint_cons)

turnover_plot(to_CDAR_mint_cons, to_CDAR)

correlation_plot(
    corrsp_CDAR_mint_cons,
    corrbond_CDAR_mint_cons,
    corrsp_CDAR,
    corrbond_CDAR,
)


stackplts(weight_CDAR_mint_cons)

t_CDAR_mint_cons = pd.concat(
    [
        performance_measures(
            weight_CDAR_mint_cons,
            aum_CDAR_mint_cons,
            returns_CDAR_mint_cons,
            to_CDAR_mint_cons,
            False,
        ),
        performance_measures(
            weight_CDAR_mint_cons,
            aum_CDAR_mint_cons_tc,
            returns_CDAR_mint_cons_tc,
            to_CDAR_mint_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_mint_cons.columns = [
    "Corr & Mint - CDAR - No costs",
    "Corr & Mint - CDAR - With costs",
]


# Optimal CDAR:

weight_CDAR_risk_mint_cons = [0] * rw_number
weight_CDAR_risk_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_number), desc="CDAR risk mint optimization under constraint"):
    weight_CDAR_risk_mint_cons[i] = pd.DataFrame(
        data=pf.CDAR_risk(
            rw_returns[i],
            np.array(weight_CDAR_risk_mint_cons[i - 1]),
            turnover_penalty,
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_CDAR_risk_mint_cons[i - 1],
            index=weight_CDAR_risk_mint_cons[i].index,
            columns=weight_CDAR_risk_mint_cons[i].columns,
        )
    else:
        w0 = weight_CDAR_risk_mint_cons[i - 1]

    length_diff = len(weight_CDAR_risk_mint_cons[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_CDAR_risk_mint_cons[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.1
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.1

del i


(
    aum_CDAR_risk_mint_cons,
    returns_CDAR_risk_mint_cons,
    to_CDAR_risk_mint_cons,
    aum_CDAR_risk_mint_cons_tc,
    returns_CDAR_risk_mint_cons_tc,
    corrsp_CDAR_risk_mint_cons,
    corrbond_CDAR_risk_mint_cons,
) = portfolio(weight_CDAR_risk_mint_cons)

turnover_plot(to_CDAR_risk_mint_cons, to_CDAR_risk)

correlation_plot(
    corrsp_CDAR_risk_mint_cons,
    corrbond_CDAR_risk_mint_cons,
    corrsp_CDAR_risk,
    corrbond_CDAR_risk,
)

stackplts(weight_CDAR_risk_mint_cons)

t_CDAR_risk_mint_cons = pd.concat(
    [
        performance_measures(
            weight_CDAR_risk_mint_cons,
            aum_CDAR_risk_mint_cons,
            returns_CDAR_risk_mint_cons,
            to_CDAR_risk_mint_cons,
            False,
        ),
        performance_measures(
            weight_CDAR_risk_mint_cons,
            aum_CDAR_risk_mint_cons_tc,
            returns_CDAR_risk_mint_cons_tc,
            to_CDAR_risk_mint_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_CDAR_risk_mint_cons.columns = [
    "Corr & Mint - CDAR risk - No costs",
    "Corr & Mint - CDAR risk - With costs",
]


###############################################################################


# Minimum risk Omega:

weight_Omegamin_mint_cons = [0] * rw_number
weight_Omegamin_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(range(rw_number), desc="Omegamin mint optimization under constraint"):
    weight_Omegamin_mint_cons[i] = pd.DataFrame(
        data=pf.Omega_min(
            rw_returns[i],
            np.array(weight_Omegamin_mint_cons[i - 1]),
            0.0002,
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i


(
    aum_Omegamin_mint_cons,
    returns_Omegamin_mint_cons,
    to_Omegamin_mint_cons,
    aum_Omegamin_mint_cons_tc,
    returns_Omegamin_mint_cons_tc,
    corrsp_Omegamin_mint_cons,
    corrbond_Omegamin_mint_cons,
) = portfolio(weight_Omegamin_mint_cons)


turnover_plot(to_Omegamin_mint_cons, to_Omegamin)

correlation_plot(
    corrsp_Omegamin_mint_cons,
    corrbond_Omegamin_mint_cons,
    corrsp_Omegamin,
    corrbond_Omegamin,
)

stackplts(weight_Omegamin_mint_cons)

t_Omegamin_mint_cons = pd.concat(
    [
        performance_measures(
            weight_Omegamin_mint_cons,
            aum_Omegamin_mint_cons,
            returns_Omegamin_mint_cons,
            to_Omegamin_mint_cons,
            False,
        ),
        performance_measures(
            weight_Omegamin_mint_cons,
            aum_Omegamin_mint_cons_tc,
            returns_Omegamin_mint_cons_tc,
            to_Omegamin_mint_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamin_mint_cons.columns = [
    "Corr & Mint - Omegamin - No costs",
    "Corr & Mint - Omegamin - With costs",
]


# Optimal Omega:

weight_Omegamax_mint_cons = [0] * rw_number
weight_Omegamax_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(range(rw_number), desc="Omegamax mint optimization under constraint"):
    weight_Omegamax_mint_cons[i] = pd.DataFrame(
        data=pf.Omega_max(
            rw_returns[i],
            np.array(weight_Omegamax_mint_cons[i - 1]),
            turnover_penalty,
            corr_sp=rw_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_Omegamax_mint_cons[i - 1],
            index=weight_Omegamax_mint_cons[i].index,
            columns=weight_Omegamax_mint_cons[i].columns,
        )
    else:
        w0 = weight_Omegamax_mint_cons[i - 1]

    length_diff = len(weight_Omegamax_mint_cons[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_Omegamax_mint_cons[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2


del i

(
    aum_Omegamax_mint_cons,
    returns_Omegamax_mint_cons,
    to_Omegamax_mint_cons,
    aum_Omegamax_mint_cons_tc,
    returns_Omegamax_mint_cons_tc,
    corrsp_Omegamax_mint_cons,
    corrbond_Omegamax_mint_cons,
) = portfolio(weight_Omegamax_mint_cons)

turnover_plot(to_Omegamax_mint_cons, to_Omegamax)


correlation_plot(
    corrsp_Omegamax_mint_cons,
    corrbond_Omegamax_mint_cons,
    corrsp_Omegamax,
    corrbond_Omegamax,
)

stackplts(weight_Omegamax_mint_cons)

t_Omegamax_mint_cons = pd.concat(
    [
        performance_measures(
            weight_Omegamax_mint_cons,
            aum_Omegamax_mint_cons,
            returns_Omegamax_mint_cons,
            to_Omegamax_mint_cons,
            False,
        ),
        performance_measures(
            weight_Omegamax_mint_cons,
            aum_Omegamax_mint_cons_tc,
            returns_Omegamax_mint_cons_tc,
            to_Omegamax_mint_cons,
            False,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_Omegamax_mint_cons.columns = [
    "Corr & Mint - Omegamax - No costs",
    "Corr & Mint - Omegamax - With costs",
]


###############################################################################

# Performance table

t_historical_P4 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_CVAR_mint_cons,
        t_CVAR_risk_mint_cons,
        t_CDAR_mint_cons,
        t_CDAR_risk_mint_cons,
        t_Omegamin_mint_cons,
        t_Omegamax_mint_cons,
    ],
    axis=1,
    ignore_index=False,
)

t_historical_P4.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_historical_P4.to_excel(writer, sheet_name="P4 - Historical")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with both cost minimization and correlation constraints computed using historical returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_CVAR_mint_cons_tc)
sns.lineplot(data=aum_CVAR_risk_mint_cons_tc)
sns.lineplot(data=aum_CDAR_mint_cons_tc)
sns.lineplot(data=aum_CDAR_risk_mint_cons_tc)
sns.lineplot(data=aum_Omegamin_mint_cons_tc)
sns.lineplot(data=aum_Omegamax_mint_cons_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P4.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with both cost minimization and correlation constraints computed using historical returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_CVAR_mint_cons)
sns.lineplot(data=aum_CVAR_risk_mint_cons)
sns.lineplot(data=aum_CDAR_mint_cons)
sns.lineplot(data=aum_CDAR_risk_mint_cons)
sns.lineplot(data=aum_Omegamin_mint_cons)
sns.lineplot(data=aum_Omegamax_mint_cons)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("historical_P4.png")


###############################################################################


#                    Minimization of costs and constraint on correlation  with synthetic returns         cons + mint + syn     #


###############################################################################


# Minimum risk cvar

weight_syn_CVAR_mint_cons = [0] * rw_corr_number
weight_syn_CVAR_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(
    range(rw_corr_number), desc="syn CVAR mint optimization under constraint"
):
    weight_syn_CVAR_mint_cons[i] = pd.DataFrame(
        data=pf.CVAR(
            rw_corr_returns[i],
            np.array(weight_syn_CVAR_mint_cons[i - 1]),
            0.0005,
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CVAR_mint_cons,
    returns_syn_CVAR_mint_cons,
    to_syn_CVAR_mint_cons,
    aum_syn_CVAR_mint_cons_tc,
    returns_syn_CVAR_mint_cons_tc,
    corrsp_syn_CVAR_mint_cons,
    corrbond_syn_CVAR_mint_cons,
) = portfolio_syn(weight_syn_CVAR_mint_cons)


turnover_plot(to_syn_CVAR_mint_cons, to_syn_CVAR)


correlation_plot(
    corrsp_syn_CVAR_mint_cons,
    corrbond_syn_CVAR_mint_cons,
    corrsp_syn_CVAR,
    corrbond_syn_CVAR,
)

stackplts(weight_syn_CVAR_mint_cons)

t_syn_CVAR_mint_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_mint_cons,
            aum_syn_CVAR_mint_cons,
            returns_syn_CVAR_mint_cons,
            to_syn_CVAR_mint_cons,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_mint_cons,
            aum_syn_CVAR_mint_cons_tc,
            returns_syn_CVAR_mint_cons_tc,
            to_syn_CVAR_mint_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_mint_cons.columns = [
    "Corr & Mint - CVAR synth. - No costs",
    "Corr & Mint - CVAR synth. - With costs",
]


# Optimal CVAR:

weight_syn_CVAR_risk_mint_cons = [0] * rw_corr_number
weight_syn_CVAR_risk_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])


turnover_penalty = 0.3
w0 = None
for i in tqdm(
    range(rw_corr_number), desc="syn CVAR risk mint optimization under constraint"
):
    weight_syn_CVAR_risk_mint_cons[i] = pd.DataFrame(
        data=pf.CVAR_risk(
            rw_corr_returns[i],
            np.array(weight_syn_CVAR_risk_mint_cons[i - 1]),
            turnover_penalty,
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_syn_CVAR_risk_mint_cons[i - 1],
            index=weight_syn_CVAR_risk_mint_cons[i].index,
            columns=weight_syn_CVAR_risk_mint_cons[i].columns,
        )
    else:
        w0 = weight_syn_CVAR_risk_mint_cons[i - 1]

    length_diff = len(weight_syn_CVAR_risk_mint_cons[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_syn_CVAR_risk_mint_cons[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2

del i


(
    aum_syn_CVAR_risk_mint_cons,
    returns_syn_CVAR_risk_mint_cons,
    to_syn_CVAR_risk_mint_cons,
    aum_syn_CVAR_risk_mint_cons_tc,
    returns_syn_CVAR_risk_mint_cons_tc,
    corrsp_syn_CVAR_risk_mint_cons,
    corrbond_syn_CVAR_risk_mint_cons,
) = portfolio_syn(weight_syn_CVAR_risk_mint_cons)

turnover_plot(to_syn_CVAR_risk_mint_cons, to_syn_CVAR_risk)

correlation_plot(
    corrsp_syn_CVAR_risk_mint_cons,
    corrbond_syn_CVAR_risk_mint_cons,
    corrsp_syn_CVAR_risk,
    corrbond_syn_CVAR_risk,
)

stackplts(weight_syn_CVAR_risk_mint_cons)

t_syn_CVAR_risk_mint_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CVAR_risk_mint_cons,
            aum_syn_CVAR_risk_mint_cons,
            returns_syn_CVAR_risk_mint_cons,
            to_syn_CVAR_risk_mint_cons,
            True,
        ),
        performance_measures(
            weight_syn_CVAR_risk_mint_cons,
            aum_syn_CVAR_risk_mint_cons_tc,
            returns_syn_CVAR_risk_mint_cons_tc,
            to_syn_CVAR_risk_mint_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CVAR_risk_mint_cons.columns = [
    "Corr & Mint - CVAR risk synth. - No costs",
    "Corr & Mint - CVAR risk synth. - With costs",
]


###############################################################################


# Minimum risk CDAR:

weight_syn_CDAR_mint_cons = [0] * rw_corr_number
weight_syn_CDAR_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(
    range(rw_corr_number), desc="syn CDAR mint optimization under constraint"
):
    weight_syn_CDAR_mint_cons[i] = pd.DataFrame(
        data=pf.CDAR(
            rw_corr_returns[i],
            np.array(weight_syn_CDAR_mint_cons[i - 1]),
            0.001,
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_CDAR_mint_cons,
    returns_syn_CDAR_mint_cons,
    to_syn_CDAR_mint_cons,
    aum_syn_CDAR_mint_cons_tc,
    returns_syn_CDAR_mint_cons_tc,
    corrsp_syn_CDAR_mint_cons,
    corrbond_syn_CDAR_mint_cons,
) = portfolio_syn(weight_syn_CDAR_mint_cons)

turnover_plot(to_syn_CDAR_mint_cons, to_syn_CDAR)


correlation_plot(
    corrsp_syn_CDAR_mint_cons,
    corrbond_syn_CDAR_mint_cons,
    corrsp_syn_CDAR,
    corrbond_syn_CDAR,
)

stackplts(weight_syn_CDAR_mint_cons)

t_syn_CDAR_mint_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_mint_cons,
            aum_syn_CDAR_mint_cons,
            returns_syn_CDAR_mint_cons,
            to_syn_CDAR_mint_cons,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_mint_cons,
            aum_syn_CDAR_mint_cons_tc,
            returns_syn_CDAR_mint_cons_tc,
            to_syn_CDAR_mint_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_mint_cons.columns = [
    "Corr & Mint - CDAR synth. - No costs",
    "Corr & Mint - CDAR synth. - With costs",
]


# Optimal CDAR:

weight_syn_CDAR_risk_mint_cons = [0] * rw_corr_number
weight_syn_CDAR_risk_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(
    range(rw_corr_number), desc="syn CDAR risk mint optimization under constraint"
):
    weight_syn_CDAR_risk_mint_cons[i] = pd.DataFrame(
        data=pf.CDAR_risk(
            rw_corr_returns[i],
            np.array(weight_syn_CDAR_risk_mint_cons[i - 1]),
            turnover_penalty,
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_syn_CDAR_risk_mint_cons[i - 1],
            index=weight_syn_CDAR_risk_mint_cons[i].index,
            columns=weight_syn_CDAR_risk_mint_cons[i].columns,
        )
    else:
        w0 = weight_syn_CDAR_risk_mint_cons[i - 1]

    length_diff = len(weight_syn_CDAR_risk_mint_cons[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_syn_CDAR_risk_mint_cons[i].values - w0.values))

    if turnov.item() > 0.5:
        turnover_penalty *= 1.2
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2


del i


(
    aum_syn_CDAR_risk_mint_cons,
    returns_syn_CDAR_risk_mint_cons,
    to_syn_CDAR_risk_mint_cons,
    aum_syn_CDAR_risk_mint_cons_tc,
    returns_syn_CDAR_risk_mint_cons_tc,
    corrsp_syn_CDAR_risk_mint_cons,
    corrbond_syn_CDAR_risk_mint_cons,
) = portfolio_syn(weight_syn_CDAR_risk_mint_cons)

turnover_plot(to_syn_CDAR_risk_mint_cons, to_syn_CDAR_risk)

correlation_plot(
    corrsp_syn_CDAR_risk_mint_cons,
    corrbond_syn_CDAR_risk_mint_cons,
    corrsp_syn_CDAR_risk,
    corrbond_syn_CDAR_risk,
)


stackplts(weight_syn_CDAR_risk_mint_cons)

t_syn_CDAR_risk_mint_cons = pd.concat(
    [
        performance_measures(
            weight_syn_CDAR_risk_mint_cons,
            aum_syn_CDAR_risk_mint_cons,
            returns_syn_CDAR_risk_mint_cons,
            to_syn_CDAR_risk_mint_cons,
            True,
        ),
        performance_measures(
            weight_syn_CDAR_risk_mint_cons,
            aum_syn_CDAR_risk_mint_cons_tc,
            returns_syn_CDAR_risk_mint_cons_tc,
            to_syn_CDAR_risk_mint_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_CDAR_risk_mint_cons.columns = [
    "Corr & Mint - CDAR risk synth. - No costs",
    "Corr & Mint - CDAR risk synth. - With costs",
]


###############################################################################


# Minimum risk Omega:

weight_syn_Omegamin_mint_cons = [0] * rw_corr_number
weight_syn_Omegamin_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

for i in tqdm(
    range(rw_corr_number), desc="syn Omegamin mint optimization under constraint"
):
    weight_syn_Omegamin_mint_cons[i] = pd.DataFrame(
        data=pf.Omega_min(
            rw_corr_returns[i],
            np.array(weight_syn_Omegamin_mint_cons[i - 1]),
            0.0002,
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i


(
    aum_syn_Omegamin_mint_cons,
    returns_syn_Omegamin_mint_cons,
    to_syn_Omegamin_mint_cons,
    aum_syn_Omegamin_mint_cons_tc,
    returns_syn_Omegamin_mint_cons_tc,
    corrsp_syn_Omegamin_mint_cons,
    corrbond_syn_Omegamin_mint_cons,
) = portfolio_syn(weight_syn_Omegamin_mint_cons)


turnover_plot(to_syn_Omegamin_mint_cons, to_syn_Omegamin)

correlation_plot(
    corrsp_syn_Omegamin_mint_cons,
    corrbond_syn_Omegamin_mint_cons,
    corrsp_syn_Omegamin,
    corrbond_syn_Omegamin,
)

stackplts(weight_syn_Omegamin_mint_cons)

t_syn_Omegamin_mint_cons = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamin_mint_cons,
            aum_syn_Omegamin_mint_cons,
            returns_syn_Omegamin_mint_cons,
            to_syn_Omegamin_mint_cons,
            True,
        ),
        performance_measures(
            weight_syn_Omegamin_mint_cons,
            aum_syn_Omegamin_mint_cons_tc,
            returns_syn_Omegamin_mint_cons_tc,
            to_syn_Omegamin_mint_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamin_mint_cons.columns = [
    "Corr & Mint - Omegamin synth. - No costs",
    "Corr & Mint - Omegamin synth. - With costs",
]


# Optimal Omega:

weight_syn_Omegamax_mint_cons = [0] * rw_corr_number
weight_syn_Omegamax_mint_cons[-1] = [0] * len(Returns.iloc[rw][rw_returns[0].columns])

turnover_penalty = 0.3
w0 = None
for i in tqdm(
    range(rw_corr_number), desc="syn Omegamax mint optimization under constraint"
):
    weight_syn_Omegamax_mint_cons[i] = pd.DataFrame(
        data=pf.Omega_max(
            rw_corr_returns[i],
            np.array(weight_syn_Omegamax_mint_cons[i - 1]),
            5,
            corr_sp=rw_corr_sp[i],
            maxcorr_sp=MaxCorrelationSP,
            corr_bond=rw_corr_bond[i],
            maxcorr_bond=MaxCorrelationBond,
        ),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

    if i == 0:
        w0 = pd.DataFrame(
            weight_syn_Omegamax_mint_cons[i - 1],
            index=weight_syn_Omegamax_mint_cons[i].index,
            columns=weight_syn_Omegamax_mint_cons[i].columns,
        )
    else:
        w0 = weight_syn_Omegamax_mint_cons[i - 1]

    length_diff = len(weight_syn_Omegamax_mint_cons[i]) - len(w0)

    if length_diff > 0:

        zeros_df = pd.DataFrame(0, index=np.arange(length_diff), columns=w0.columns)
        w0 = pd.concat([w0, zeros_df])

    turnov = np.sum(np.abs(weight_syn_Omegamax_mint_cons[i].values - w0.values))

    if turnov.item() > 0.4:
        turnover_penalty *= 1.3
    elif turnov.item() < 0.2:
        turnover_penalty /= 1.2

del i


(
    aum_syn_Omegamax_mint_cons,
    returns_syn_Omegamax_mint_cons,
    to_syn_Omegamax_mint_cons,
    aum_syn_Omegamax_mint_cons_tc,
    returns_syn_Omegamax_mint_cons_tc,
    corrsp_syn_Omegamax_mint_cons,
    corrbond_syn_Omegamax_mint_cons,
) = portfolio_syn(weight_syn_Omegamax_mint_cons)

turnover_plot(to_syn_Omegamax_mint_cons, to_syn_Omegamax)

correlation_plot(
    corrsp_syn_Omegamax_mint_cons,
    corrbond_syn_Omegamax_mint_cons,
    corrsp_syn_Omegamax,
    corrbond_syn_Omegamax,
)

stackplts(weight_syn_Omegamax_mint_cons)

t_syn_Omegamax_mint_cons = pd.concat(
    [
        performance_measures(
            weight_syn_Omegamax_mint_cons,
            aum_syn_Omegamax_mint_cons,
            returns_syn_Omegamax_mint_cons,
            to_syn_Omegamax_mint_cons,
            True,
        ),
        performance_measures(
            weight_syn_Omegamax_mint_cons,
            aum_syn_Omegamax_mint_cons_tc,
            returns_syn_Omegamax_mint_cons_tc,
            to_syn_Omegamax_mint_cons,
            True,
        ),
    ],
    axis=1,
    ignore_index=True,
)
t_syn_Omegamax_mint_cons.columns = [
    "Corr & Mint - Omegamax synth. - No costs",
    "Corr & Mint - Omegamax synth. - With costs",
]

###############################################################################


# Performance table

t_syn_P4 = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_MSR,
        t_syn_CVAR_mint_cons,
        t_syn_CVAR_risk_mint_cons,
        t_syn_CDAR_mint_cons,
        t_syn_CDAR_risk_mint_cons,
        t_syn_Omegamin_mint_cons,
        t_syn_Omegamax_mint_cons,
    ],
    axis=1,
    ignore_index=False,
)

t_syn_P4.columns = [
    "FoF",
    "EW",
    "EW - With costs",
    "MVP",
    "MVP - With costs",
    "MSR",
    "MSR - With costs",
    "Min. risk CVaR",
    "Min. risk CVaR - With costs",
    "Optimal CVaR",
    "Optimal CVaR - With costs",
    "Min. risk CDaR",
    "Min. risk CDaR - With costs",
    "Optimal CDaR",
    "Optimal CDaR - With costs",
    "Min. risk Omega",
    "Min. risk Omega - With costs",
    "Optimal Omega",
    "Optimal Omega - With costs",
]

with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_syn_P4.to_excel(writer, sheet_name="P4 - Synthetic")


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with  both cost minimization and correlation constraints computed using synthetic returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW_tc, linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=aum_MVP_tc, linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=aum_MSR_tc, linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=aum_syn_CVAR_mint_cons_tc)
sns.lineplot(data=aum_syn_CVAR_risk_mint_cons_tc)
sns.lineplot(data=aum_syn_CDAR_mint_cons_tc)
sns.lineplot(data=aum_syn_CDAR_risk_mint_cons_tc)
sns.lineplot(data=aum_syn_Omegamin_mint_cons_tc)
sns.lineplot(data=aum_syn_Omegamax_mint_cons_tc)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 900)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P4.png")


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with both cost minimization and correlation constraints computed using synthetic returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=aum_EW, linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=aum_MVP, linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=aum_MSR, linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=aum_syn_CVAR_mint_cons)
sns.lineplot(data=aum_syn_CVAR_risk_mint_cons)
sns.lineplot(data=aum_syn_CDAR_mint_cons)
sns.lineplot(data=aum_syn_CDAR_risk_mint_cons)
sns.lineplot(data=aum_syn_Omegamin_mint_cons)
sns.lineplot(data=aum_syn_Omegamax_mint_cons)
plt.autoscale(tight="x")
plt.ylabel("Cumulative Returns (%)")
plt.xlabel("Time")
plt.ylim(0, 1350)
plt.legend(
    [
        "Fund of Fund index",
        "Equally Weighted",
        "Minimum Variance",
        "Maximum Sharpe ratio",
        "Minimum risk CVAR",
        "Optimal CVAR",
        "Minimum risk CDAR",
        "Optimal CDAR",
        "Minimum risk Omega",
        "Optimal Omega",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=5,
)
plt.savefig("synthetic_P4.png")

###############################################################################


#                             Summary Tables                                  #


###############################################################################


t_minrisk = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MVP,
        t_CVAR,
        t_CDAR,
        t_Omegamin,
        t_syn_CVAR,
        t_syn_CDAR,
        t_syn_Omegamin,
        t_CVAR_mint,
        t_CDAR_mint,
        t_Omegamin_mint,
        t_syn_CVAR_mint,
        t_syn_CDAR_mint,
        t_syn_Omegamin_mint,
        t_CVAR_cons,
        t_CDAR_cons,
        t_Omegamin_cons,
        t_syn_CVAR_cons,
        t_syn_CDAR_cons,
        t_syn_Omegamin_cons,
        t_CVAR_mint_cons,
        t_CDAR_mint_cons,
        t_Omegamin_mint_cons,
        t_syn_CVAR_mint_cons,
        t_syn_CDAR_mint_cons,
        t_syn_Omegamin_mint_cons,
    ],
    axis=1,
    ignore_index=False,
)


t_optimal = pd.concat(
    [
        t_FoF,
        t_EW,
        t_MSR,
        t_CVAR_risk,
        t_CDAR_risk,
        t_Omegamax,
        t_syn_CVAR_risk,
        t_syn_CDAR_risk,
        t_syn_Omegamax,
        t_CVAR_risk_mint,
        t_CDAR_risk_mint,
        t_Omegamax_mint,
        t_syn_CVAR_risk_mint,
        t_syn_CDAR_risk_mint,
        t_syn_Omegamax_mint,
        t_CVAR_risk_cons,
        t_CDAR_risk_cons,
        t_Omegamax_cons,
        t_syn_CVAR_risk_cons,
        t_syn_CDAR_risk_cons,
        t_syn_Omegamax_cons,
        t_CVAR_risk_mint_cons,
        t_CDAR_risk_mint_cons,
        t_Omegamax_mint_cons,
        t_syn_CVAR_risk_mint_cons,
        t_syn_CDAR_risk_mint_cons,
        t_syn_Omegamax_mint_cons,
    ],
    axis=1,
    ignore_index=False,
)


with pd.ExcelWriter("Performance Measures.xlsx") as writer:
    t_minrisk.to_excel(writer, sheet_name="Conservative")
    t_optimal.to_excel(writer, sheet_name="Aggressive")
del writer


###############################################################################
