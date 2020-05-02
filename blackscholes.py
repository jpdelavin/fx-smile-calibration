import numpy as np
from scipy.stats import norm
from scipy import optimize


def forward_price(spot, mty, rd, rf):
    """
    Computes the forward price

    Parameters
    ----------
    spot : float
        Spot price
    mty : float
        Time to maturity (in years)
    rd : float
        Domestic currency interest rate per annum
    rf : float
        Foreign currency interest rate per annum

    Returns
    -------
    float
        No-arbitrage forward price
    """

    return spot * np.exp(mty * (rd - rf))


def d12(fwd, strike, mty, vol):
    """
    Calculates d1 and d2 in the Black-Scholes model

    Parameters
    ----------
    fwd : float
        Forward price
    strike : float
        Strike price of the option
    mty : float
        Maturity of the option in years
    vol : float
        Volatility of the returns of the underlying

    Returns
    -------
    (float, float)
        d1 and d2 in the Black-Scholes model
    """

    first_term = np.log(fwd / strike) / (vol * np.sqrt(mty))
    second_term = 0.5 * vol * np.sqrt(mty)

    return first_term + second_term, first_term - second_term


def option_price(price, strike, mty, rd, rf, vol, option_type="c", price_type="s"):
    """
    Prices an option according to the Black-Scholes model

    Parameters
    ----------
    price : float
        Spot or forward price of the underlying
    strike : float
        Strike price
    mty : float
        Time to maturity (in years)
    rd : float
        Domestic currency interest rate per annum
    rf : float
        Foreign currency interest rate per annum
    vol : float
        Volatility of the return of the underlying per annum
    option_type : str, optional
        "c", if the option to be priced is a call
        "p", if the option to be priced is a put
    price_type : str, optional
        "s", if price given is the spot price
        "f", if price given is the forward price

    Returns
    -------
    float
        Price of the option according to the Black-Scholes model
    """

    assert option_type.lower() in ["c", "p"], "Invalid option_type given"
    assert price_type.lower() in ["s", "f"], "Invalid price_type given"

    if price_type.lower() == "s":
        forward = forward_price(price, mty, rd, rf)
    else:
        forward = price

    if option_type.lower() == "c":
        phi = 1
    else:
        phi = -1

    d1, d2 = d12(forward, strike, mty, vol)
    value = phi * np.exp(-rd * mty) * (forward * norm.cdf(phi * d1) - strike * norm.cdf(phi * d2))

    return value


def delta_from_strike_and_vol(price, strike, mty, rd, rf, vol, option_type="c", price_type="s",
                              delta_type="f", premium_included="i"):
    """
    Computes the option delta according to the Black-Scholes model

    Parameters
    ----------
    price : float
        Spot or forward price of the underlying
    strike : float
        Strike price
    mty : float
        Time to maturity (in years)
    rd : float
        Domestic currency interest rate per annum
    rf : float
        Foreign currency interest rate per annum
    vol : float
        Volatility of the return of the underlying per annum
    option_type : str, optional
        "c", if the option to be priced is a call
        "p", if the option to be priced is a put
    price_type : str, optional
        "s", if price given is the spot price
        "f", if price given is the forward price
    delta_type : str, optional
        "s", if delta to be computed is spot delta
        "f", if delta to be computed is forward delta
    premium_included : str, optional
        "i", if delta to be computed includes spot/forward
        "e", if delta to be computed excludes spot/forward

    Returns
    -------
    float
        Delta of the option according to the Black-Scholes model
    """

    assert option_type.lower() in ["c", "p"], "Invalid option_type given"
    assert price_type.lower() in ["s", "f"], "Invalid price_type given"
    assert delta_type.lower() in ["s", "f"], "Invalid delta_type given"
    assert premium_included.lower() in ["i", "e"], "Invalid premium_included given"

    if price_type.lower() == "s":
        forward = forward_price(price, mty, rd, rf)
    else:
        forward = price

    if option_type.lower() == "c":
        phi = 1
    else:
        phi = -1

    d1, d2 = d12(forward, strike, mty, vol)

    if delta_type.lower() == "s":
        multiplier = np.exp(-rf * mty)
    else:
        multiplier = 1

    if premium_included.lower() == "i":
        return phi * multiplier * (strike / forward) * norm.cdf(phi * d2)
    else:
        return phi * multiplier * norm.cdf(phi * d1)


def atm_dns_strike(price, mty, rd, rf, vol, price_type="s", premium_included="i"):
    """
    Computes the ATM delta-neutral strike according to the Black-Scholes model

    Parameters
    ----------
    price : float
        Spot or forward price of the underlying
    mty : float
        Time to maturity (in years)
    rd : float
        Domestic currency interest rate per annum
    rf : float
        Foreign currency interest rate per annum
    vol : float
        Volatility of the return of the underlying per annum
    price_type : str, optional
        "s", if price given is the spot price
        "f", if price given is the forward price
    premium_included : str, optional
        "i", if delta to be computed includes spot/forward
        "e", if delta to be computed excludes spot/forward

    Returns
    -------
    float
        Delta-neutral strike according to the Black-Scholes model
    """

    assert price_type.lower() in ["s", "f"], "Invalid price_type given"
    assert premium_included.lower() in ["i", "e"], "Invalid premium_included given"

    if price_type.lower() == "s":
        forward = forward_price(price, mty, rd, rf)
    else:
        forward = price

    if premium_included.lower() == "i":
        sign = -1
    else:
        sign = 1

    return forward * np.exp(0.5 * sign * mty * vol ** 2)


def strike_from_delta_and_vol(price, delta, mty, rd, rf, vol, option_type="c", price_type="s",
                              delta_type="f", premium_included="i", tol=1.5e-8):
    """
    Computes the strike for a given delta and volatility according to the Black-Scholes model

    Parameters
    ----------
    price : float
        Spot or forward price of the underlying
    delta : float
        Delta of the option
    mty : float
        Time to maturity (in years)
    rd : float
        Domestic currency interest rate per annum
    rf : float
        Foreign currency interest rate per annum
    vol : float
        Volatility of the return of the underlying per annum
    option_type : str, optional
        "c", if the option to be priced is a call
        "p", if the option to be priced is a put
    price_type : str, optional
        "s", if price given is the spot price
        "f", if price given is the forward price
    delta_type : str, optional
        "s", if delta to be computed is spot delta
        "f", if delta to be computed is forward delta
    premium_included : str, optional
        "i", if delta to be computed includes spot/forward
        "e", if delta to be computed excludes spot/forward
    tol : float, optional
        Tolerance to be used in calibration

    Returns
    -------
    float
        Delta-neutral strike according to the Black-Scholes model
    """

    assert option_type.lower() in ["c", "p"], "Invalid option_type given"
    assert price_type.lower() in ["s", "f"], "Invalid price_type given"
    assert delta_type.lower() in ["s", "f"], "Invalid delta_type given"
    assert premium_included.lower() in ["i", "e"], "Invalid premium_included given"

    if price_type.lower() == "s":
        forward = forward_price(price, mty, rd, rf)
        forward = forward_price(price, mty, rd, rf)
    else:
        forward = price

    if option_type.lower() == "c":
        phi = 1
    else:
        phi = -1
    assert delta * phi > 0, "Delta={d:.4f} given has incorrect sign.".format(d=delta)

    if delta_type.lower() == "s":
        multiplier = np.exp(-rf * mty)
    else:
        multiplier = 1

    if premium_included.lower() == "e":
        assert 0 <= delta * phi / multiplier <= 1,\
            "Delta={d:.4f} given outside allowable range of [{min:.4f}, {max:.4f}]".format(d=delta,
                                                                                           min=min(0, multiplier/phi),
                                                                                           max=max(0, multiplier/phi))
        return forward * np.exp(0.5 * mty * vol ** 2 -
                                phi * vol * np.sqrt(mty) * norm.ppf(delta * phi / multiplier))

    else:
        def delta_diff(strike):
            # Computes difference between desired delta and delta for a given strike
            return delta_from_strike_and_vol(price=price, strike=strike, mty=mty, rd=rd, rf=rf, vol=vol,
                                             option_type=option_type, price_type=price_type, delta_type=delta_type,
                                             premium_included=premium_included) - delta

        def gamma(strike):
            # First derivative of the delta
            _, d2 = d12(fwd=forward, strike=strike, mty=mty, vol=vol)
            return (multiplier / forward) * (phi * norm.cdf(phi * d2) -
                                             norm.pdf(phi * d2) / (vol * np.sqrt(mty)))

        if option_type.lower() == "p":
            # Since put premium-included delta is monotonic in strike and boundless,
            # Newton's method is guaranteed to converge
            solution = optimize.newton(func=delta_diff, fprime=gamma, x0=forward, tol=tol, rtol=tol)

        else:
            # Since there are appropriate upper and lower bounds for the strike (Reiswich & Wystup, 2009),
            # Brent is appropriate

            # Upper bound is the strike if delta were premium excluded

            assert 0 <= delta * phi / multiplier <= 1, "Delta={d:.4f} given too high.".format(d=delta)
            k_max = forward * np.exp(0.5 * mty * vol ** 2 -
                                     phi * vol * np.sqrt(mty) * norm.ppf(delta * phi / multiplier))

            # Lower bound is the strike that gives the maximum delta
            k_min = optimize.brentq(f=gamma, a=tol, b=k_max, xtol=tol, rtol=tol)

            if type(k_min) == np.ndarray:
                k_min = k_min[0]

            assert delta_diff(k_min) > 0, "Delta={d:.4f} given too high. ".format(d=delta) + \
                "Maximum possible delta is {m:.4f}.".format(m=delta_diff(k_min) + delta)

            solution = optimize.brentq(f=delta_diff, a=k_min, b=k_max, xtol=tol, rtol=tol)

        if type(solution) == np.ndarray:
            solution = solution[0]

        return solution
