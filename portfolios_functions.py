import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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


#                           Portfolios creation                               #


###############################################################################

# Function needed to define portfolios algorythms


def cvar(returns, alpha):
    """
    Compute the Conditional Value at Risk (CVaR) of returns.
    alpha = 1- confidence level

    Parameters:
    returns (ndarray): Array of returns.
    alpha (float): Significance level (0 < alpha < 1).

    Returns:
    float: CVaR of returns.
    """

    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    index = int(np.floor(alpha * n))

    cvar = -np.mean(sorted_returns[:index])

    return cvar


def cdar(returns, alpha):
    """
    Compute the Conditional Drawdown at Risk (CDaR) of returns.
    alpha = 1- confidence level

    Parameters:
    returns (ndarray): Array of returns.
    alpha (float): Significance level (0 < alpha < 1).

    Returns:
    float: CDaR of returns.
    """
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max

    sorted_drawdowns = np.sort(drawdowns)
    n = len(sorted_drawdowns)
    index = int(np.floor(alpha * n))

    # Avoid empty slice or all-zero drawdowns
    if index == 0 or np.all(sorted_drawdowns[:index] == 0):
        return 1e-6  # small non-zero value to avoid division by zero

    cdar = -np.mean(sorted_drawdowns[:index])

    return cdar


def omega_ratio(returns, threshold):
    """
    Calculate the Omega ratio of a portfolio.

    Parameters:
    returns (numpy array): Array of portfolio returns.
    threshold (float): Threshold return.

    Returns:
    float: Omega ratio.
    """

    excess_returns = returns - threshold

    gains = excess_returns[returns - threshold >= 0]
    losses = -excess_returns[threshold - returns > 0]

    positive_area = np.mean(gains)
    negative_area = np.mean(losses)

    # If there are no losses, return infinity
    if negative_area == 0:
        return np.inf

    # Calculate the Omega ratio
    omega = positive_area / negative_area

    return omega


def omega_denum(returns, mar):
    """
    Calculate the denominator of the omega ratio.

    Parameters:
    returns (ndarray): Array of returns.
    mar (float): Minimum acceptable return (MAR) or threshold return.

    Returns:
    float: Omega denumerator.
    """
    negative_returns = returns[returns < mar]
    negative_weighted_average = -np.mean(negative_returns)

    denum = negative_weighted_average

    return denum


###############################################################################

# Portfolios declined into conservative and aggressive per unit of risk:

# Minimum risk portfolios (Minimize risk):


def MVP(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Compute the Minimum variance portfolio.

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones(nbrstocks) / nbrstocks
    covariance = stcks.cov()

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        obj_value = (x.T @ covariance @ x) ** 0.5

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def CVAR(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Conditional Value at risk (CVAR or Expected Shortfall), minimize average monthly loss at 95%

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones(nbrstocks) / nbrstocks

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        obj_value = cvar(portfolio_ret, 0.05)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def CDAR(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Conditional Drawdown at risk (CDAR), minimize average monthly drawdown at 95%

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones((nbrstocks)) / nbrstocks

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        obj_value = cdar(portfolio_ret, 0.05)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def Omega_min(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Minimize Omega denumerator (minimize average return under rf)

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones((nbrstocks)) / nbrstocks

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        obj_value = omega_denum(portfolio_ret, rf)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


###############################################################################

# Optimal portfolios (Maximize return per unit of risk)


def MV_risk(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Maximize return over volatility

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones((nbrstocks)) / nbrstocks
    covariance = stcks.cov()

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        obj_value = -(portfolio_ret[-1] - rf) / ((x.T @ covariance @ x) ** 0.5)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def CVAR_risk(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Maximize return over conditional value at risk

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones((nbrstocks)) / nbrstocks

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        cvar_value = cvar(portfolio_ret, alpha=0.05)
        obj_value = -((portfolio_ret[-1] - rf) / cvar_value)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - prev_weights))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def CDAR_risk(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Maximize return over conditional drawdown at risk

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones((nbrstocks)) / nbrstocks

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        cdar_value = cdar(portfolio_ret, alpha=0.05)
        obj_value = -((portfolio_ret[-1] - rf) / cdar_value)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def Omega_max(
    stcks,
    prev_weights=None,
    turnover_penalty=0.0,
    corr_sp=None,
    maxcorr_sp=1.0,
    corr_bond=None,
    maxcorr_bond=1.0,
):
    """
    Maximize omega ratio

    Parameters:
    stcks (DataFrame): Dataframe of returns.
    prev_weights (ndarray): Array of previous weight neccessary for turnover minimization.
    turnover_penalty (float): Factor on turnover penalty in objective function

    Returns:
    ndarray: Array of weights.
    """
    nbrstocks = len(stcks.transpose())

    # Optimization parameters
    ones = np.ones((nbrstocks, 1))
    weight_equal = np.ones((nbrstocks)) / nbrstocks

    def portfolio_return(weights):
        return np.dot(stcks.values, weights)

    # Constraint on sum of weights equal one
    constraints = [{"type": "eq", "fun": lambda x: np.dot(ones.T, x) - 1}]
    if corr_sp is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_sp) + maxcorr_sp}
        )
    if corr_bond is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: -np.sum(x * corr_bond) + maxcorr_bond}
        )

    # Objective function
    def objective(x):
        portfolio_ret = portfolio_return(x)
        obj_value = -omega_ratio(portfolio_ret, rf)

        if prev_weights is not None:
            w0 = prev_weights
            length_diff = len(x) - len(w0)

            if length_diff > 0:
                w0 = np.vstack([w0, np.array([[0]] * length_diff)])

            turnover = np.sum(np.abs(x - w0))
            obj_value += turnover_penalty * turnover
        return obj_value

    result = minimize(
        lambda x: objective(x),
        weight_equal,
        method="SLSQP",
        tol=1e-10,
        # Max individual weight of 50% to force diversification
        bounds=[(0, 0.5) for _ in range(nbrstocks)],
        constraints=constraints,
    )
    return result.x


def init_weights_mint(rw_returns):
    weights = [0] * len(rw_returns)
    weights[0] = np.ones(len(rw_returns[0].columns)) / len(rw_returns[0].columns)
    return weights


class PortfolioData:
    def __init__(
        self,
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
    ):
        self.Returns = Returns
        self.rw = rw
        self.rw_returns = rw_returns
        self.rw_number = rw_number
        self.rw_corr_number = rw_corr_number
        self.rw_sp = rw_sp
        self.rw_bond = rw_bond
        self.rw_corr_sp = rw_corr_sp
        self.rw_corr_bond = rw_corr_bond
        self.bench = bench
        self.returns_bench = returns_bench
        self.returns_syn_bench = returns_syn_bench


# FIXME: Default transaction cost 30bps
class Portfolio:
    def __init__(self, weight, data: PortfolioData, tc=0.003, syn=False, name=None):
        """
        Initialize Portfolio with weights, returns, transaction cost, and related data.
        """
        self.weight = weight
        self.data = data
        self.tc = tc
        self.syn = syn
        self.name = name or "Unnamed Portfolio"

        # Set useful indices once
        self.index_full = self.data.Returns.iloc[self.data.rw - 1 :].index
        self.index_short = self.data.Returns.iloc[self.data.rw :].index
        self.index_correlation = self.data.Returns.iloc[
            self.data.rw : self.data.rw + self.data.rw_number
        ].index

        self.index_full_syn = self.data.Returns.iloc[self.data.rw :].index
        self.index_short_syn = self.data.Returns.iloc[self.data.rw + 1 :].index
        self.index_correlation_syn = self.data.Returns.iloc[
            self.data.rw + 1 : self.data.rw + self.data.rw_corr_number + 1
        ].index

    def compute_aum(self):
        aum = pd.Series(
            data=[100] * (len(self.weight) + 1),
            index=self.index_full_syn if self.syn else self.index_full,
            name="AUM",
            dtype=float,
        )

        for i, w in enumerate(self.weight):
            columns_idx = self.data.Returns.columns.get_indexer(
                self.data.rw_returns[i].columns
            )
            returns_row = self.data.Returns.iloc[self.data.rw + i, columns_idx]
            aum.iloc[i + 1] = (1 + w.T @ returns_row) * aum.iloc[i]
        return aum

    def compute_return(self):
        returns = pd.Series(
            data=[0] * (len(self.weight) + 1),
            index=self.index_full_syn if self.syn else self.index_full,
            name="Return",
            dtype=float,
        )
        for i, w in enumerate(self.weight):
            columns_idx = self.data.Returns.columns.get_indexer(
                self.data.rw_returns[i].columns
            )
            returns_row = self.data.Returns.iloc[self.data.rw + i, columns_idx]
            returns.iloc[i + 1] = w.T @ returns_row
        return returns

    def compute_turnover(self):
        to = pd.Series(
            data=[0] * len(self.weight),
            index=self.index_short_syn if self.syn else self.index_short,
            name="Turnover",
            dtype=float,
        )
        for i in range(len(self.weight) - 1):
            columns_idx = self.data.Returns.columns.get_indexer(
                self.data.rw_returns[i].columns
            )
            returns_row = self.data.Returns.iloc[self.data.rw + i, columns_idx]
            w1 = self.weight[i + 1].squeeze().values
            w0 = (self.weight[i].squeeze() * (1 + returns_row)).values

            length_diff = len(w1) - len(w0)
            if length_diff > 0:
                w0 = np.pad(w0, (0, length_diff), mode="constant")

            to.iloc[i + 1] = abs(w1 - w0).sum()
        return to

    def compute_aum_tc(self):
        turnover = self.compute_turnover()
        aum_tc = pd.Series(
            data=[100] * (len(self.weight) + 1),
            index=self.index_full_syn if self.syn else self.index_full,
            name="AUM",
            dtype=float,
        )
        for i, w in enumerate(self.weight):
            columns_idx = self.data.Returns.columns.get_indexer(
                self.data.rw_returns[i].columns
            )
            returns_row = self.data.Returns.iloc[self.data.rw + i, columns_idx]
            net_return = (w.T @ returns_row) - turnover.iloc[i] * self.tc
            aum_tc.iloc[i + 1] = (1 + net_return) * aum_tc.iloc[i]
        return aum_tc

    def compute_return_tc(self):
        turnover = self.compute_turnover()
        returns_tc = pd.Series(
            data=[0] * (len(self.weight) + 1),
            index=self.index_full_syn if self.syn else self.index_full,
            name="Return",
            dtype=float,
        )
        for i, w in enumerate(self.weight):
            columns_idx = self.data.Returns.columns.get_indexer(
                self.data.rw_returns[i].columns
            )
            returns_row = self.data.Returns.iloc[self.data.rw + i, columns_idx]
            returns_tc.iloc[i + 1] = (w.T @ returns_row) - turnover.iloc[i] * self.tc
        return returns_tc

    def compute_correlation(self, corr_data):
        correl = pd.Series(
            data=[np.nan] * self.data.rw_number,
            index=self.index_correlation_syn if self.syn else self.index_correlation,
            name="Correlation",
            dtype=float,
        )
        for i in range(self.data.rw_number):
            correl.iloc[i] = self.weight[i].mul(corr_data[i], axis=0).sum()
        return correl

    def compute_correlation_syn(self, corr_data):
        correl = pd.Series(
            data=[np.nan] * self.data.rw_corr_number,
            index=self.index_correlation_syn if self.syn else self.index_correlation,
            name="Syn_Correlation",
            dtype=float,
        )

        for i in range(self.data.rw_corr_number):
            correl.iloc[i] = self.weight[i].mul(corr_data[i], axis=0).sum()

        return correl

    def run(self):
        """
        Compute all portfolio series and return them
        """

        aum = self.compute_aum()
        returns = self.compute_return()
        turnover = self.compute_turnover()
        aum_tc = self.compute_aum_tc()
        returns_tc = self.compute_return_tc()

        if self.syn:
            correl_sp = self.compute_correlation_syn(self.data.rw_corr_sp)
            correl_bond = self.compute_correlation_syn(self.data.rw_corr_bond)

        else:
            correl_sp = self.compute_correlation(self.data.rw_sp)
            correl_bond = self.compute_correlation(self.data.rw_bond)

        return aum, returns, turnover, aum_tc, returns_tc, correl_sp, correl_bond

    def _information_ratio(self, returns, benchmark_returns):
        excess_returns = returns - benchmark_returns
        average_excess_return = np.mean(excess_returns)
        tracking_error = np.std(excess_returns, ddof=1)
        return average_excess_return / tracking_error

    def _max_drawdown(self, aum):
        n = len(aum)
        peak = aum.iloc[0]
        max_dd = 0.0
        for i in range(1, n):
            if aum.iloc[i] > peak:
                peak = aum.iloc[i]
            else:
                drawdown = (peak - aum.iloc[i]) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
        return max_dd

    def _r_squared(self, returns, index_returns):
        returns = np.array(returns).reshape(-1, 1)
        index_returns = np.array(index_returns).reshape(-1, 1)
        model = LinearRegression()
        model.fit(index_returns, returns)
        return model.score(index_returns, returns)

    def compute_performance(self, tc=False):
        """
        Computes performance measures and returns a pandas Series.
        """

        if tc:
            aum = self.compute_aum_tc()
            returns = self.compute_return_tc()

        else:
            aum = self.compute_aum()
            returns = self.compute_return()

        to = self.compute_turnover()

        AR = (1 + returns.mean()) ** 12 - 1
        SD = returns.std() * np.sqrt(12)
        MDD = self._max_drawdown(aum)
        CVaR = cvar(returns, 0.01)  # Note: you need to ensure 'pf' is available.
        CDaR = cdar(returns, 0.01)  # Same here
        SR = (AR - Rf_mean) / SD
        M_squared = Rf_mean + SR * self.data.bench.iloc[self.data.rw :].std() * np.sqrt(
            12
        )
        Calmar = (AR - Rf_mean) / MDD

        if self.syn:
            R_squared = self._r_squared(returns, self.data.returns_syn_bench)
            CORR_SP = self.compute_correlation_syn(self.data.rw_corr_sp).mean()
            CORR_BOND = self.compute_correlation_syn(self.data.rw_corr_bond).mean()
            CORR_bench = returns.corr(self.data.bench.iloc[self.data.rw + 1 :])
        else:
            R_squared = self._r_squared(returns, self.data.returns_bench)
            CORR_SP = self.compute_correlation(self.data.rw_sp).mean()
            CORR_BOND = self.compute_correlation(self.data.rw_bond).mean()
            CORR_bench = returns.corr(self.data.bench.iloc[self.data.rw :])

        PT = to.sum()

        measure = pd.Series(
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
        return measure

    def compute_table(self):
        table = pd.concat(
            [
                self.compute_performance(tc=False),
                self.compute_performance(tc=True),
            ],
            axis=1,
            ignore_index=True,
        )
        table.columns = [f"{self.name} - No costs", f"{self.name} - With costs"]

        return table

    def stackplts(self):
        weight_frames = []  # Create a list to collect weight DataFrames

        for i in range(len(self.weight)):
            weight_frames.append(self.weight[i].T)

        data = pd.concat(weight_frames)  # Only concatenate ONCE after the loop

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.stackplot(
            data.index,
            [data[col].fillna(0) for col in self.data.Returns.columns],
            labels=self.data.Returns.columns,
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
        ax.set_title(f"Weight allocation: {self.name} portfolio")
        fig.savefig(f"Graphs/Stackplots/{self.name}.png")
        plt.close(fig)

    def turnover_plot(self, to_comp):
        """
        Outputs a graph plotting comparison between portfolios with and without costs minimization
        """
        to = self.compute_turnover()

        # Create a new figure and axes for each plot
        fig, ax = plt.subplots(figsize=(16, 9))

        to_comp.plot(
            kind="area",
            legend=None,
            ax=ax,
            color="tab:blue",
            label="No minimization: Total To: " + str(round(to_comp.sum(), 2)),
        )
        to.plot(
            kind="area",
            ax=ax,
            color="tab:red",
            label="With minimization: Total To: " + str(round(to.sum(), 2)),
        )
        plt.title(
            f"Turnover comparison: {self.name} portfolio w. & w/ cost minimization "
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
        plt.savefig(f"Graphs/Turnover/{self.name}_costs.png")
        plt.close(fig)

    def correlation_plot(self, corrsp_comp, corrbond_comp):
        """
        Outputs a graph plotting correlation of a portfolio to the S&P and bond index overtime
        """
        syn = self.syn

        if syn:
            corrsp = self.compute_correlation_syn(self.data.rw_corr_sp)
            corrbond = self.compute_correlation_syn(self.data.rw_corr_bond)

        else:
            corrsp = self.compute_correlation(self.data.rw_sp)
            corrbond = self.compute_correlation(self.data.rw_bond)

        # Create a new figure and axes for each plot
        fig, ax = plt.subplots(figsize=(16, 9))

        corrsp.plot(legend=None, ax=ax, color="tab:blue", linestyle="--")
        corrbond.plot(ax=ax, color="tab:red", linestyle="--")
        corrsp_comp.plot(ax=ax, color="tab:blue", linestyle="dotted")
        corrbond_comp.plot(ax=ax, color="tab:red", linestyle="dotted")

        plt.title(f"Correlation to stocks and bonds:{self.name} portfolio")
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
        plt.savefig(f"Graphs/Correlation/{self.name}_correlations.png")
        plt.close(fig)


### Evaluation function helper


def compare_objects(obj1, obj2, rtol=1e-8, atol=1e-10):
    """
    Compare two pandas Series or DataFrames and print:
    - .equals result
    - np.allclose result
    - Maximum absolute difference
    """

    print("=== Comparison Report ===")

    # 1. Strict equality
    print(f"Strict equality result: {obj1.equals(obj2)}")

    # 2. allclose (allowing for floating point tolerance)
    try:
        allclose_result = np.allclose(obj1.values, obj2.values, rtol=rtol, atol=atol)
        print(f"Equality with floating point tolerance result: {allclose_result}")
    except Exception as e:
        print(f"np.allclose() error: {e}")

    # 3. Maximum absolute difference
    try:
        max_diff = (obj1 - obj2).abs().max()
        print(f"Maximum absolute difference: {max_diff}")
    except Exception as e:
        print(f"Max diff calculation error: {e}")

    print("==========================")
