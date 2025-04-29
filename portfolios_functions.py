import pandas as pd
import numpy as np
from scipy.optimize import minimize


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

