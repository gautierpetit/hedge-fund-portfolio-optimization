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
    "Data/HFRI_full.xlsx",
    sheet_name="Data",
    index_col=0,
    parse_dates=True,
    date_format="%b-%y",
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
plt.close()

Price.plot(figsize=(16, 9))
plt.autoscale(tight="x")
plt.ylabel("Index Price")
plt.xlabel("Time")
plt.ylim(0, 30000)
plt.legend(Price.columns, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5)
plt.title("Prices of of HFRI indices")
plt.savefig("Price.png")
plt.close()


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


# Get Fund of Fund index returns and get them to proper size to work with rw
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


###############################################################################
###############################################################################
###############################################################################


#                           Bechmark Portfolios                               #


###############################################################################
###############################################################################
###############################################################################
###############################################################################

data = pf.PortfolioData(
    Returns,
    rw,
    rw_returns,
    rw_number,
    rw_corr_number,
    rw_sp,
    rw_bond,
    rw_corr_sp,
    rw_corr_bond,
    bench,
    returns_bench,
    returns_syn_bench,
)


t_FoF = pd.read_excel("Data/HFRI_full.xlsx", sheet_name="FoF", index_col=0)

# Equally weighted portfolio

weight_EW = [0] * rw_number

for i in tqdm(range(rw_number), desc="EW optimization"):
    weight_EW[i] = pd.DataFrame(
        data=[1 / len(rw_returns[i].T)] * len(rw_returns[i].T),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )


del i

EW = pf.Portfolio(weight_EW, data, name="EW")
EW.stackplts()
t_EW = EW.compute_table()

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

MVP = pf.Portfolio(weight_MVP, data, name="MVP")
MVP.stackplts()
t_MVP = MVP.compute_table()

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


MSR = pf.Portfolio(weight_MSR, data, name="MSR")
MSR.stackplts()
t_MSR = MSR.compute_table()

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


sns_plot = plt.figure(figsize=(16, 9))
plt.title("Benchmark Portfolio Wealth (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
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
plt.close()


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

CVAR = pf.Portfolio(weight_CVAR, data, name="CVAR")
CVAR.stackplts()
t_CVAR = CVAR.compute_table()

# Optimal CVAR:

weight_CVAR_risk = [0] * rw_number

for i in tqdm(range(rw_number), desc="CVAR risk optimization"):
    weight_CVAR_risk[i] = pd.DataFrame(
        data=pf.CVAR_risk(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i

CVAR_risk = pf.Portfolio(weight_CVAR_risk, data, name="CVAR risk")
CVAR_risk.stackplts()
t_CVAR_risk = CVAR_risk.compute_table()

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


CDAR = pf.Portfolio(weight_CDAR, data, name="CDAR")
CDAR.stackplts()
t_CDAR = CDAR.compute_table()

# Optimal CDAR:

weight_CDAR_risk = [0] * rw_number

for i in tqdm(range(rw_number), desc="CDAR risk optimization"):
    weight_CDAR_risk[i] = pd.DataFrame(
        data=pf.CDAR_risk(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i

CDAR_risk = pf.Portfolio(weight_CDAR_risk, data, name="CDAR risk")
CDAR_risk.stackplts()
t_CDAR_risk = CDAR_risk.compute_table()

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

Omegamin = pf.Portfolio(weight_Omegamin, data, name="Omega denum.")
Omegamin.stackplts()
t_Omegamin = Omegamin.compute_table()

# Optimal Omega:

weight_Omegamax = [0] * rw_number

for i in tqdm(range(rw_number), desc="Omegamax optimization"):
    weight_Omegamax[i] = pd.DataFrame(
        data=pf.Omega_max(rw_returns[i]),
        columns=Returns.iloc[i + rw : i + rw + 1].index,
        index=rw_returns[i].columns,
    )

del i

Omegamax = pf.Portfolio(weight_Omegamax, data, name="Omega ratio")
Omegamax.stackplts()
t_Omegamax = Omegamax.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using historical returns with costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=CVAR.compute_aum_tc())
sns.lineplot(data=CVAR_risk.compute_aum_tc())
sns.lineplot(data=CDAR.compute_aum_tc())
sns.lineplot(data=CDAR_risk.compute_aum_tc())
sns.lineplot(data=Omegamin.compute_aum_tc())
sns.lineplot(data=Omegamax.compute_aum_tc())
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
plt.savefig("historical_P1_costs.png")
plt.close()


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using historical returns without costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=CVAR.compute_aum())
sns.lineplot(data=CVAR_risk.compute_aum())
sns.lineplot(data=CDAR.compute_aum())
sns.lineplot(data=CDAR_risk.compute_aum())
sns.lineplot(data=Omegamin.compute_aum())
sns.lineplot(data=Omegamax.compute_aum())
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
plt.close()

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

syn_CVAR = pf.Portfolio(weight_syn_CVAR, data, syn=True, name="CVAR synth.")
syn_CVAR.stackplts()
t_syn_CVAR = syn_CVAR.compute_table()

# Optimal CVAR:

weight_syn_CVAR_risk = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CVAR risk optimization"):
    weight_syn_CVAR_risk[i] = pd.DataFrame(
        data=pf.CVAR_risk(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i

syn_CVAR_risk = pf.Portfolio(
    weight_syn_CVAR_risk, data, syn=True, name="CVAR risk synth."
)
syn_CVAR_risk.stackplts()
t_syn_CVAR_risk = syn_CVAR_risk.compute_table()

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

syn_CDAR = pf.Portfolio(weight_syn_CDAR, data, syn=True, name="CDAR synth.")
syn_CDAR.stackplts()
t_syn_CDAR = syn_CDAR.compute_table()

# Optimal CDAR:

weight_syn_CDAR_risk = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn CDAR risk optimization"):
    weight_syn_CDAR_risk[i] = pd.DataFrame(
        data=pf.CDAR_risk(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i

syn_CDAR_risk = pf.Portfolio(
    weight_syn_CDAR_risk, data, syn=True, name="CDAR risk synth."
)
syn_CDAR_risk.stackplts()
t_syn_CDAR_risk = syn_CDAR_risk.compute_table()

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

syn_Omegamin = pf.Portfolio(
    weight_syn_Omegamin, data, syn=True, name="Omega denum. synth."
)
syn_Omegamin.stackplts()
t_syn_Omegamin = syn_Omegamin.compute_table()

# Optimal Omega:

weight_syn_Omegamax = [0] * rw_corr_number

for i in tqdm(range(rw_corr_number), desc="syn Omegamax optimization"):
    weight_syn_Omegamax[i] = pd.DataFrame(
        data=pf.Omega_max(rw_corr_returns[i]),
        columns=Returns.iloc[i + rw + 1 : i + rw + 2].index,
        index=rw_corr_returns[i].columns,
    )

del i

syn_Omegamax = pf.Portfolio(
    weight_syn_Omegamax, data, syn=True, name="Omega ratio synth."
)
syn_Omegamax.stackplts()
t_syn_Omegamax = syn_Omegamax.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using synthetic returns with costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=syn_CVAR.compute_aum_tc())
sns.lineplot(data=syn_CVAR_risk.compute_aum_tc())
sns.lineplot(data=syn_CDAR.compute_aum_tc())
sns.lineplot(data=syn_CDAR_risk.compute_aum_tc())
sns.lineplot(data=syn_Omegamin.compute_aum_tc())
sns.lineplot(data=syn_Omegamax.compute_aum_tc())
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
plt.savefig("synthetic_P1_costs.png")
plt.close()


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title("Portfolio computed using historical returns without costs (base level=100)")
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=syn_CVAR.compute_aum())
sns.lineplot(data=syn_CVAR_risk.compute_aum())
sns.lineplot(data=syn_CDAR.compute_aum())
sns.lineplot(data=syn_CDAR_risk.compute_aum())
sns.lineplot(data=syn_Omegamin.compute_aum())
sns.lineplot(data=syn_Omegamax.compute_aum())
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
plt.close()


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


CVAR_mint = pf.Portfolio(weight_CVAR_mint, data, name="Min t - CVAR")
CVAR_mint.stackplts()
CVAR_mint.turnover_plot(CVAR.compute_turnover())
t_CVAR_mint = CVAR_mint.compute_table()

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

CVAR_risk_mint = pf.Portfolio(weight_CVAR_risk_mint, data, name="Min t - CVAR")
CVAR_risk_mint.stackplts()
CVAR_risk_mint.turnover_plot(CVAR_risk.compute_turnover())
t_CVAR_risk_mint = CVAR_risk_mint.compute_table()

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

CDAR_mint = pf.Portfolio(weight_CDAR_mint, data, name="Min t - CDAR")
CDAR_mint.stackplts()
CDAR_mint.turnover_plot(CDAR.compute_turnover())
t_CDAR_mint = CDAR_mint.compute_table()

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


CDAR_risk_mint = pf.Portfolio(weight_CDAR_risk_mint, data, name="Min t - CDAR")
CDAR_risk_mint.stackplts()
CDAR_risk_mint.turnover_plot(CDAR_risk.compute_turnover())
t_CDAR_risk_mint = CDAR_risk_mint.compute_table()

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


Omegamin_mint = pf.Portfolio(weight_Omegamin_mint, data, name="Min t - Omega denum.")
Omegamin_mint.stackplts()
Omegamin_mint.turnover_plot(Omegamin.compute_turnover())
t_Omegamin_mint = Omegamin_mint.compute_table()

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


Omegamax_mint = pf.Portfolio(weight_Omegamax_mint, data, name="Min t - Omega ratio")
Omegamax_mint.stackplts()
Omegamax_mint.turnover_plot(Omegamax.compute_turnover())
t_Omegamax_mint = Omegamax_mint.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using historical returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=CVAR_mint.compute_aum_tc())
sns.lineplot(data=CVAR_risk_mint.compute_aum_tc())
sns.lineplot(data=CDAR_mint.compute_aum_tc())
sns.lineplot(data=CDAR_risk_mint.compute_aum_tc())
sns.lineplot(data=Omegamin_mint.compute_aum_tc())
sns.lineplot(data=Omegamax_mint.compute_aum_tc())
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
plt.savefig("historical_P2_costs.png")
plt.close()


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using historical returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=CVAR_mint.compute_aum())
sns.lineplot(data=CVAR_risk_mint.compute_aum())
sns.lineplot(data=CDAR_mint.compute_aum())
sns.lineplot(data=CDAR_risk_mint.compute_aum())
sns.lineplot(data=Omegamin_mint.compute_aum())
sns.lineplot(data=Omegamax_mint.compute_aum())
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
plt.close()


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

syn_CVAR_mint = pf.Portfolio(
    weight_syn_CVAR_mint, data, syn=True, name="Min t - CVAR synth."
)
syn_CVAR_mint.stackplts()
syn_CVAR_mint.turnover_plot(syn_CVAR.compute_turnover())
t_syn_CVAR_mint = syn_CVAR_mint.compute_table()

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

syn_CVAR_risk_mint = pf.Portfolio(
    weight_syn_CVAR_risk_mint, data, syn=True, name="Min t - CVAR synth."
)
syn_CVAR_risk_mint.stackplts()
syn_CVAR_risk_mint.turnover_plot(syn_CVAR_risk.compute_turnover())
t_syn_CVAR_risk_mint = syn_CVAR_risk_mint.compute_table()

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

syn_CDAR_mint = pf.Portfolio(
    weight_syn_CDAR_mint, data, syn=True, name="Min t - CDAR synth."
)
syn_CDAR_mint.stackplts()
syn_CDAR_mint.turnover_plot(syn_CDAR.compute_turnover())
t_syn_CDAR_mint = syn_CDAR_mint.compute_table()

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

syn_CDAR_risk_mint = pf.Portfolio(
    weight_syn_CDAR_risk_mint, data, syn=True, name="Min t - CDAR synth."
)
syn_CDAR_risk_mint.stackplts()
syn_CDAR_risk_mint.turnover_plot(syn_CDAR_risk.compute_turnover())
t_syn_CDAR_risk_mint = syn_CDAR_risk_mint.compute_table()

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

syn_Omegamin_mint = pf.Portfolio(
    weight_syn_Omegamin_mint, data, syn=True, name="Min t - Omega denum. synth"
)
syn_Omegamin_mint.stackplts()
syn_Omegamin_mint.turnover_plot(syn_Omegamin.compute_turnover())
t_syn_Omegamin_mint = syn_Omegamin_mint.compute_table()

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

syn_Omegamax_mint = pf.Portfolio(
    weight_syn_Omegamax_mint, data, syn=True, name="Min t - Omega ratio synth."
)
syn_Omegamax_mint.stackplts()
syn_Omegamax_mint.turnover_plot(syn_Omegamax.compute_turnover())
t_syn_Omegamax_mint = syn_Omegamax_mint.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using synthetic returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=syn_CVAR_mint.compute_aum_tc())
sns.lineplot(data=syn_CVAR_risk_mint.compute_aum_tc())
sns.lineplot(data=syn_CDAR_mint.compute_aum_tc())
sns.lineplot(data=syn_CDAR_risk_mint.compute_aum_tc())
sns.lineplot(data=syn_Omegamin_mint.compute_aum_tc())
sns.lineplot(data=syn_Omegamax_mint.compute_aum_tc())
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
plt.savefig("synthetic_P2_costs.png")
plt.close()


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using synthetic returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=syn_CVAR_mint.compute_aum())
sns.lineplot(data=syn_CVAR_risk_mint.compute_aum())
sns.lineplot(data=syn_CDAR_mint.compute_aum())
sns.lineplot(data=syn_CDAR_risk_mint.compute_aum())
sns.lineplot(data=syn_Omegamin_mint.compute_aum())
sns.lineplot(data=syn_Omegamax_mint.compute_aum())
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
plt.close()


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

CVAR_cons = pf.Portfolio(weight_CVAR_cons, data, name="Corr - CVAR")
CVAR_cons.stackplts()
CVAR_cons.correlation_plot(
    CVAR.compute_correlation(rw_sp), CVAR.compute_correlation(rw_bond)
)
t_CVAR_cons = CVAR_cons.compute_table()

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

CVAR_risk_cons = pf.Portfolio(weight_CVAR_risk_cons, data, name="Corr - CVAR Risk")
CVAR_risk_cons.stackplts()
CVAR_risk_cons.correlation_plot(
    CVAR_risk.compute_correlation(rw_sp), CVAR_risk.compute_correlation(rw_bond)
)
t_CVAR_risk_cons = CVAR_risk_cons.compute_table()

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

CDAR_cons = pf.Portfolio(weight_CDAR_cons, data, name="Corr - CDAR")
CDAR_cons.stackplts()
CDAR_cons.correlation_plot(
    CDAR.compute_correlation(rw_sp), CDAR.compute_correlation(rw_bond)
)
t_CDAR_cons = CDAR_cons.compute_table()

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

CDAR_risk_cons = pf.Portfolio(weight_CDAR_risk_cons, data, name="Corr - CDAR Risk")
CDAR_risk_cons.stackplts()
CDAR_risk_cons.correlation_plot(
    CDAR_risk.compute_correlation(rw_sp), CDAR_risk.compute_correlation(rw_bond)
)
t_CDAR_risk_cons = CDAR_risk_cons.compute_table()

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

Omegamin_cons = pf.Portfolio(weight_Omegamin_cons, data, name="Corr - Omegamin")
Omegamin_cons.stackplts()
Omegamin_cons.correlation_plot(
    Omegamin.compute_correlation(rw_sp), Omegamin.compute_correlation(rw_bond)
)
t_Omegamin_cons = Omegamin_cons.compute_table()

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

Omegamax_cons = pf.Portfolio(weight_Omegamax_cons, data, name="Corr - Omegamax")
Omegamax_cons.stackplts()
Omegamax_cons.correlation_plot(
    Omegamax.compute_correlation(rw_sp), Omegamax.compute_correlation(rw_bond)
)
t_Omegamax_cons = Omegamax_cons.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with correlation constraints computed using historical returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=CVAR_cons.compute_aum_tc())
sns.lineplot(data=CVAR_risk_cons.compute_aum_tc())
sns.lineplot(data=CDAR_cons.compute_aum_tc())
sns.lineplot(data=CDAR_risk_cons.compute_aum_tc())
sns.lineplot(data=Omegamin_cons.compute_aum_tc())
sns.lineplot(data=Omegamax_cons.compute_aum_tc())
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
plt.savefig("historical_P3_costs.png")
plt.close()


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with correlation constraints computed using historical returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=CVAR_cons.compute_aum())
sns.lineplot(data=CVAR_risk_cons.compute_aum())
sns.lineplot(data=CDAR_cons.compute_aum())
sns.lineplot(data=CDAR_risk_cons.compute_aum())
sns.lineplot(data=Omegamin_cons.compute_aum())
sns.lineplot(data=Omegamax_cons.compute_aum())
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
plt.close()


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

syn_CVAR_cons = pf.Portfolio(
    weight_syn_CVAR_cons, data, syn=True, name="Corr - CVAR synth."
)
syn_CVAR_cons.stackplts()
syn_CVAR_cons.correlation_plot(
    syn_CVAR.compute_correlation_syn(rw_corr_sp),
    syn_CVAR.compute_correlation_syn(rw_corr_bond),
)
t_syn_CVAR_cons = syn_CVAR_cons.compute_table()

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

syn_CVAR_risk_cons = pf.Portfolio(
    weight_syn_CVAR_risk_cons, data, syn=True, name="Corr - CVAR risk synth."
)
syn_CVAR_risk_cons.stackplts()
syn_CVAR_risk_cons.correlation_plot(
    syn_CVAR_risk.compute_correlation_syn(rw_corr_sp),
    syn_CVAR_risk.compute_correlation_syn(rw_corr_bond),
)
t_syn_CVAR_risk_cons = syn_CVAR_risk_cons.compute_table()

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

syn_CDAR_cons = pf.Portfolio(
    weight_syn_CDAR_cons, data, syn=True, name="Corr - CDAR synth."
)
syn_CDAR_cons.stackplts()
syn_CDAR_cons.correlation_plot(
    syn_CDAR.compute_correlation_syn(rw_corr_sp),
    syn_CDAR.compute_correlation_syn(rw_corr_bond),
)
t_syn_CDAR_cons = syn_CDAR_cons.compute_table()

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

syn_CDAR_risk_cons = pf.Portfolio(
    weight_syn_CDAR_risk_cons, data, syn=True, name="Corr - CDAR synth."
)
syn_CDAR_risk_cons.stackplts()
syn_CDAR_risk_cons.correlation_plot(
    syn_CDAR_risk.compute_correlation_syn(rw_corr_sp),
    syn_CDAR_risk.compute_correlation_syn(rw_corr_bond),
)
t_syn_CDAR_risk_cons = syn_CDAR_risk_cons.compute_table()

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

syn_Omegamin_cons = pf.Portfolio(
    weight_syn_Omegamin_cons, data, syn=True, name="Corr - Omegamin synth."
)
syn_Omegamin_cons.stackplts()
syn_Omegamin_cons.correlation_plot(
    syn_Omegamin.compute_correlation_syn(rw_corr_sp),
    syn_Omegamin.compute_correlation_syn(rw_corr_bond),
)
t_syn_Omegamin_cons = syn_Omegamin_cons.compute_table()

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

syn_Omegamax_cons = pf.Portfolio(
    weight_syn_Omegamax_cons, data, syn=True, name="Corr - Omegamax synth."
)
syn_Omegamax_cons.stackplts()
syn_Omegamax_cons.correlation_plot(
    syn_Omegamax.compute_correlation_syn(rw_corr_sp),
    syn_Omegamax.compute_correlation_syn(rw_corr_bond),
)
t_syn_Omegamax_cons = syn_Omegamax_cons.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with correlation constraints computed using synthetic returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=syn_CVAR_cons.compute_aum_tc())
sns.lineplot(data=syn_CVAR_risk_cons.compute_aum_tc())
sns.lineplot(data=syn_CDAR_cons.compute_aum_tc())
sns.lineplot(data=syn_CDAR_risk_cons.compute_aum_tc())
sns.lineplot(data=syn_Omegamin_cons.compute_aum_tc())
sns.lineplot(data=syn_Omegamax_cons.compute_aum_tc())
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
plt.savefig("synthetic_P3_costs.png")
plt.close()


# Without costs:
sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with cost optimized computed using synthetic returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=syn_CVAR_cons.compute_aum())
sns.lineplot(data=syn_CVAR_risk_cons.compute_aum())
sns.lineplot(data=syn_CDAR_cons.compute_aum())
sns.lineplot(data=syn_CDAR_risk_cons.compute_aum())
sns.lineplot(data=syn_Omegamin_cons.compute_aum())
sns.lineplot(data=syn_Omegamax_cons.compute_aum())
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
plt.close()

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


CVAR_mint_cons = pf.Portfolio(weight_CVAR_mint_cons, data, name="Corr & Mint - CVAR")
CVAR_mint_cons.stackplts()
CVAR_mint_cons.turnover_plot(CVAR.compute_turnover())
CVAR_mint_cons.correlation_plot(
    CVAR.compute_correlation(rw_sp), CVAR.compute_correlation(rw_bond)
)
t_CVAR_mint_cons = CVAR_mint_cons.compute_table()

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


CVAR_risk_mint_cons = pf.Portfolio(
    weight_CVAR_risk_mint_cons, data, name="Corr & Mint - CVAR risk"
)
CVAR_risk_mint_cons.stackplts()
CVAR_risk_mint_cons.turnover_plot(CVAR_risk.compute_turnover())
CVAR_risk_mint_cons.correlation_plot(
    CVAR_risk.compute_correlation(rw_sp), CVAR_risk.compute_correlation(rw_bond)
)
t_CVAR_risk_mint_cons = CVAR_risk_mint_cons.compute_table()

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

CDAR_mint_cons = pf.Portfolio(weight_CDAR_mint_cons, data, name="Corr & Mint - CDAR")
CDAR_mint_cons.stackplts()
CDAR_mint_cons.turnover_plot(CDAR.compute_turnover())
CDAR_mint_cons.correlation_plot(
    CDAR.compute_correlation(rw_sp), CDAR.compute_correlation(rw_bond)
)
t_CDAR_mint_cons = CDAR_mint_cons.compute_table()

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


CDAR_risk_mint_cons = pf.Portfolio(
    weight_CDAR_risk_mint_cons, data, name="Corr & Mint - CDAR Risk"
)
CDAR_risk_mint_cons.stackplts()
CDAR_risk_mint_cons.turnover_plot(CDAR_risk.compute_turnover())
CDAR_risk_mint_cons.correlation_plot(
    CDAR_risk.compute_correlation(rw_sp), CDAR_risk.compute_correlation(rw_bond)
)
t_CDAR_risk_mint_cons = CDAR_risk_mint_cons.compute_table()

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


Omegamin_mint_cons = pf.Portfolio(
    weight_Omegamin_mint_cons, data, name="Corr & Mint - Omegamin"
)
Omegamin_mint_cons.stackplts()
Omegamin_mint_cons.turnover_plot(Omegamin.compute_turnover())
Omegamin_mint_cons.correlation_plot(
    Omegamin.compute_correlation(rw_sp), Omegamin.compute_correlation(rw_bond)
)
t_Omegamin_mint_cons = Omegamin_mint_cons.compute_table()

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

Omegamax_mint_cons = pf.Portfolio(
    weight_Omegamax_mint_cons, data, name="Corr & Mint - Omegamax"
)
Omegamax_mint_cons.stackplts()
Omegamax_mint_cons.turnover_plot(Omegamax.compute_turnover())
Omegamax_mint_cons.correlation_plot(
    Omegamax.compute_correlation(rw_sp), Omegamax.compute_correlation(rw_bond)
)
t_Omegamax_mint_cons = Omegamax_mint_cons.compute_table()

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


# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with both cost minimization and correlation constraints computed using historical returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=CVAR_mint_cons.compute_aum_tc())
sns.lineplot(data=CVAR_risk_mint_cons.compute_aum_tc())
sns.lineplot(data=CDAR_mint_cons.compute_aum_tc())
sns.lineplot(data=CDAR_risk_mint_cons.compute_aum_tc())
sns.lineplot(data=Omegamin_mint_cons.compute_aum_tc())
sns.lineplot(data=Omegamax_mint_cons.compute_aum_tc())
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
plt.savefig("historical_P4_costs.png")
plt.close()


# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with both cost minimization and correlation constraints computed using historical returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=CVAR_mint_cons.compute_aum())
sns.lineplot(data=CVAR_risk_mint_cons.compute_aum())
sns.lineplot(data=CDAR_mint_cons.compute_aum())
sns.lineplot(data=CDAR_risk_mint_cons.compute_aum())
sns.lineplot(data=Omegamin_mint_cons.compute_aum())
sns.lineplot(data=Omegamax_mint_cons.compute_aum())
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
plt.close()

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

syn_CVAR_mint_cons = pf.Portfolio(
    weight_syn_CVAR_mint_cons, data, syn=True, name="Corr & Mint - CVAR synth."
)
syn_CVAR_mint_cons.stackplts()
syn_CVAR_mint_cons.turnover_plot(syn_CVAR.compute_turnover())
syn_CVAR_mint_cons.correlation_plot(
    syn_CVAR.compute_correlation_syn(rw_corr_sp),
    syn_CVAR.compute_correlation_syn(rw_corr_bond),
)
t_syn_CVAR_mint_cons = syn_CVAR_mint_cons.compute_table()

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

syn_CVAR_risk_mint_cons = pf.Portfolio(
    weight_syn_CVAR_risk_mint_cons,
    data,
    syn=True,
    name="Corr & Mint - CVAR Risk synth.",
)
syn_CVAR_risk_mint_cons.stackplts()
syn_CVAR_risk_mint_cons.turnover_plot(syn_CVAR_risk.compute_turnover())
syn_CVAR_risk_mint_cons.correlation_plot(
    syn_CVAR_risk.compute_correlation_syn(rw_corr_sp),
    syn_CVAR_risk.compute_correlation_syn(rw_corr_bond),
)
t_syn_CVAR_risk_mint_cons = syn_CVAR_risk_mint_cons.compute_table()

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

syn_CDAR_mint_cons = pf.Portfolio(
    weight_syn_CDAR_mint_cons, data, syn=True, name="Corr & Mint - CDAR synth."
)
syn_CDAR_mint_cons.stackplts()
syn_CDAR_mint_cons.turnover_plot(syn_CDAR.compute_turnover())
syn_CDAR_mint_cons.correlation_plot(
    syn_CDAR.compute_correlation_syn(rw_corr_sp),
    syn_CDAR.compute_correlation_syn(rw_corr_bond),
)
t_syn_CDAR_mint_cons = syn_CDAR_mint_cons.compute_table()

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

syn_CDAR_risk_mint_cons = pf.Portfolio(
    weight_syn_CDAR_risk_mint_cons,
    data,
    syn=True,
    name="Corr & Mint - CDAR Risk synth.",
)
syn_CDAR_risk_mint_cons.stackplts()
syn_CDAR_risk_mint_cons.turnover_plot(syn_CDAR_risk.compute_turnover())
syn_CDAR_risk_mint_cons.correlation_plot(
    syn_CDAR_risk.compute_correlation_syn(rw_corr_sp),
    syn_CDAR_risk.compute_correlation_syn(rw_corr_bond),
)
t_syn_CDAR_risk_mint_cons = syn_CDAR_risk_mint_cons.compute_table()

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

syn_Omegamin_mint_cons = pf.Portfolio(
    weight_syn_Omegamin_mint_cons, data, syn=True, name="Corr & Mint - Omegamin synth."
)
syn_Omegamin_mint_cons.stackplts()
syn_Omegamin_mint_cons.turnover_plot(syn_Omegamin.compute_turnover())
syn_Omegamin_mint_cons.correlation_plot(
    syn_Omegamin.compute_correlation_syn(rw_corr_sp),
    syn_Omegamin.compute_correlation_syn(rw_corr_bond),
)
t_syn_Omegamin_mint_cons = syn_Omegamin_mint_cons.compute_table()

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

syn_Omegamax_mint_cons = pf.Portfolio(
    weight_syn_Omegamax_mint_cons, data, syn=True, name="Corr & Mint - Omegamax synth."
)
syn_Omegamax_mint_cons.stackplts()
syn_Omegamax_mint_cons.turnover_plot(syn_Omegamax.compute_turnover())
syn_Omegamax_mint_cons.correlation_plot(
    syn_Omegamax.compute_correlation_syn(rw_corr_sp),
    syn_Omegamax.compute_correlation_syn(rw_corr_bond),
)
t_syn_Omegamax_mint_cons = syn_Omegamax_mint_cons.compute_table()

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

# With costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with  both cost minimization and correlation constraints computed using synthetic returns with costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum_tc(), linestyle=(0, (1, 1)), color="dimgrey")
sns.lineplot(data=MVP.compute_aum_tc(), linestyle=(0, (5, 5)), color="dimgrey")
sns.lineplot(data=MSR.compute_aum_tc(), linestyle=(0, (5, 1)), color="dimgrey")
sns.lineplot(data=syn_CVAR_mint_cons.compute_aum_tc())
sns.lineplot(data=syn_CVAR_risk_mint_cons.compute_aum_tc())
sns.lineplot(data=syn_CDAR_mint_cons.compute_aum_tc())
sns.lineplot(data=syn_CDAR_risk_mint_cons.compute_aum_tc())
sns.lineplot(data=syn_Omegamin_mint_cons.compute_aum_tc())
sns.lineplot(data=syn_Omegamax_mint_cons.compute_aum_tc())
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
plt.savefig("synthetic_P4_costs.png")
plt.close()

# Without costs:

sns.set_palette("tab10")
sns_plot = plt.figure(figsize=(16, 9))
plt.title(
    "Portfolio with both cost minimization and correlation constraints computed using synthetic returns without costs (base level=100)"
)
sns.lineplot(data=aum_FoF, linestyle=(0, (1, 5)), color="dimgrey")
sns.lineplot(data=EW.compute_aum(), linestyle=(0, (1, 1)), color="black")
sns.lineplot(data=MVP.compute_aum(), linestyle=(0, (5, 5)), color="black")
sns.lineplot(data=MSR.compute_aum(), linestyle=(0, (5, 1)), color="black")
sns.lineplot(data=syn_CVAR_mint_cons.compute_aum())
sns.lineplot(data=syn_CVAR_risk_mint_cons.compute_aum())
sns.lineplot(data=syn_CDAR_mint_cons.compute_aum())
sns.lineplot(data=syn_CDAR_risk_mint_cons.compute_aum())
sns.lineplot(data=syn_Omegamin_mint_cons.compute_aum())
sns.lineplot(data=syn_Omegamax_mint_cons.compute_aum())
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
plt.close()

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
    t_bench.to_excel(writer, sheet_name="Benchmark")
    t_historical_P1.to_excel(writer, sheet_name="P1 - Historical")
    t_syn_P1.to_excel(writer, sheet_name="P1 - Synthetic")
    t_historical_P2.to_excel(writer, sheet_name="P2 - Historical")
    t_syn_P2.to_excel(writer, sheet_name="P2 - Synthetic")
    t_historical_P3.to_excel(writer, sheet_name="P3 - Historical")
    t_syn_P3.to_excel(writer, sheet_name="P3 - Synthetic")
    t_historical_P4.to_excel(writer, sheet_name="P4 - Historical")
    t_syn_P4.to_excel(writer, sheet_name="P4 - Synthetic")
    t_minrisk.to_excel(writer, sheet_name="Conservative")
    t_optimal.to_excel(writer, sheet_name="Aggressive")
###############################################################################
