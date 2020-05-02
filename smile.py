import numpy as np
from scipy import optimize
import inspect

import blackscholes

import warnings
warnings.filterwarnings('ignore')


class SmileCalibration:
    """
    Calibrates FX volatility smile for a given curve model based on FX market convention

    Parameters
    ----------
    implied_vol_func : function
        Function that accepts the following required arguments: strike, forward, mty, and fit_params.
        Function computes the implied volatility at a given strike at the given set of parameters fit_params.
        fit_params are the parameters to be fitted by SmileCalibration.
    vol_atm : float
        ATM implied volatility quote
    vol_rr : float
        Risk reversal implied volatility quote
    vol_bf : float
        Butterfly implied volatility quote
    price : float
        Spot or forward price of the underlying
    rd : float
        Domestic currency interest rate per annum
    rf : float
        Foreign currency interest rate per annum
    mty : float
        Time to maturity in years
    rr_delta : float, optional
        Delta of the options in the risk reversal quote
    bf_delta : float, optional
        Delta of the options in the butterfly quote
    price_type : str, optional
        "s", if price given is the spot price
        "f", if price given is the forward price
    delta_type : str, optional
        "s", if delta to be used is spot delta
        "f", if delta to be used is forward delta
    premium_included : str, optional
        "i", if delta to be computed includes spot/forward
        "e", if delta to be computed excludes spot/forward
    atm_type : str, optional
        "d", if atm strike convention is delta-neutral strike
        "f", if atm strike convention is forward price
    tol : float, optional
        Tolerance to be used in calibration
    transform_params_func : func, optional
        Function that maps any real-valued set of parameters into
        the allowable range of values for the parameters.
        For example, if a parameter x can only be positive,
        abs(x) or x**2 will map any real-valued x to a positive value.
        This is used to

    Attributes
    ----------
    spot, forward : float
        Spot and forward price of the underlying
    k_rr_call, k_rr_put, vol_rr_call, vol_rr_put : float
        Fitted strikes and vols for the risk reversal quote
    k_bf_call, k_bf_put, vol_bf_call, vol_bf_put : float
        Fitted strikes and vols for the butterfly quote
    fitted : bool
        True if the model parameters have already been fitted
    fitted_params : list(float)
        List of curve parameters that fit the given implied volatility quotes
    """

    def __init__(self, implied_vol_func, vol_atm, vol_rr, vol_bf, price, rd, rf, mty, rr_delta=0.25, bf_delta=0.25,
                 price_type="s", delta_type="s", premium_included="i", atm_type="d", tol=1e-08,
                 transform_params_func=None):

        assert delta_type.lower() in ["s", "f"], "Invalid delta_type given"
        assert premium_included.lower() in ["i", "e"], "Invalid premium_included given"
        self.premium_included = premium_included.lower()
        self.delta_type = delta_type.lower()

        func_args = inspect.getfullargspec(implied_vol_func).args
        assert all([arg in func_args for arg in ["strike", "forward", "mty", "fit_params"]]), \
            "implied_vol_func does not have the proper required arguments"
        self._implied_vol_func = implied_vol_func

        if transform_params_func is not None:
            self._transform_params_func = transform_params_func
        else:
            # No transformation
            self._transform_params_func = lambda params: params

        self.vol_atm = vol_atm
        self.vol_rr = vol_rr
        self.vol_bf = vol_bf

        self.rd = rd
        self.rf = rf

        self.mty = mty

        self.tol = tol
        self.rr_delta = rr_delta
        self.bf_delta = bf_delta

        assert price_type.lower() in ["s", "f"], "Invalid price_type given"
        if price_type == "s":
            self.forward = blackscholes.forward_price(price, mty, rd, rf)
            self.spot = price
        else:
            self.spot = blackscholes.forward_price(price, -mty, rd, rf)
            self.forward = price

        assert atm_type.lower() in ["f", "d"], "Invalid atm_type given"
        self.atm_type = atm_type.lower()
        if self.atm_type == "d":
            self.k_atm = blackscholes.atm_dns_strike(price=self.forward, mty=mty, rd=rd, rf=rf, vol=self.vol_atm,
                                                     price_type="f", premium_included=self.premium_included)
        else:
            self.k_atm = self.forward

        self.k_rr_call, self.k_rr_put, self.vol_rr_call, self.vol_rr_put = None, None, None, None
        self.k_bf_call, self.k_bf_put, self.vol_bf_call, self.vol_bf_put = None, None, None, None
        self.fitted_params = None
        self.fitted = False
        self.strikes_dict, self.vols_dict = None, None

    def implied_vol(self, strike, params=None):
        """
        Computes the implied volatility for a given strike for a given set of model parameters

        Parameters
        ----------
        strike : float
            Strike price
        params : list(float), optional
            Model parameters to be used in the curve.
            If None, fitted parameters will be used in the computation.

        Returns
        -------
        float
            Implied volatility from the curve on the given strike
        """

        if params is None:
            assert self.fitted, "No params given or model params are not yet fitted"
            params = self.fitted_params

        return self._implied_vol_func(strike=strike, forward=self.forward, mty=self.mty,
                                      fit_params=params)

    def option_price(self, strike, vol, option_type):
        """
        Computes the price of the option for a given strike and vol according to the Black-Scholes model

        Parameters
        ----------
        strike : float
            Strike price
        vol : float
            Volatility of the return of the underlying per annum
        option_type : str
            "c", if the option to be priced is a call
            "p", if the option to be priced is a put

        Returns
        -------
        float
            Option price
        """

        return blackscholes.option_price(price=self.forward, strike=strike, mty=self.mty, rd=self.rd, rf=self.rf,
                                         vol=vol, option_type=option_type, price_type="f")

    def delta(self, strike, vol, option_type):
        """
        Computes the delta of the option for a given strike and vol according to the Black-Scholes model

        Parameters
        ----------
        strike : float
            Strike price
        vol : float
            Volatility of the return of the underlying per annum
        option_type : str
            "c", if the option to be priced is a call
            "p", if the option to be priced is a put

        Returns
        -------
        float
            Option delta
        """

        return blackscholes.delta_from_strike_and_vol(price=self.forward, strike=strike, mty=self.mty,
                                                      rd=self.rd, rf=self.rf, vol=vol,
                                                      option_type=option_type, price_type="f",
                                                      delta_type=self.delta_type,
                                                      premium_included=self.premium_included)

    def strike_from_delta_and_vol(self, delta, vol, option_type):
        """
        Get strike that satisfies Black-Scholes model with given delta and volatility

        Parameters
        ----------
        option_type : str
            "c", if the option to be priced is a call
            "p", if the option to be priced is a put
        delta : float
            Required delta of the option
        vol : float
            Required implied volatility of the option

        Returns
        -------
        float
            Strike that satisfies given delta and vol levels
        """

        return blackscholes.strike_from_delta_and_vol(price=self.forward, delta=delta, mty=self.mty,
                                                      rd=self.rd, rf=self.rf, vol=vol,
                                                      option_type=option_type, price_type="f",
                                                      delta_type=self.delta_type,
                                                      premium_included=self.premium_included)

    def fit_strike_from_delta(self, option_type, delta, params=None):
        """
        Get strike that satisfies Black-Scholes model with given delta and
        volatility from the implied vol curve generated by given set of parameters

        Parameters
        ----------
        option_type : str
            "c", if the option to be priced is a call
            "p", if the option to be priced is a put
        delta : float
            Required delta of the option
        params : list(float), optional
            Model parameters that will generate an implied vol curve.
            If None, fitted parameters will be used in the computation.

        Returns
        -------
        float
            Strike that satisfies given delta and implied vol from given parameters
        """

        if params is None:
            assert self.fitted, "No params given or model params are not yet fitted"
            params = self.fitted_params

        def fit_k(strike):
            return self.strike_from_delta_and_vol(option_type=option_type, delta=delta,
                                                  vol=self.implied_vol(strike=strike, params=params))

        solution = optimize.fixed_point(fit_k, x0=[self.k_atm], xtol=self.tol)

        if type(solution) == np.ndarray:
            solution = solution[0]

        return solution

    def fit(self, initial_guess):
        """
        Fits the model parameters to the given implied volatility quotes

        Parameters
        ----------
        initial_guess : list(float)
            Initial guess for the parameters

        Returns
        -------
        scipy.optimize.OptimizeResult
            Results of calibration
        """

        assert type(initial_guess) == list, "Initial guess should be a list of floats."

        # k_rr_call, k_rr_put, k_bf_call, k_bf_put, fit_params
        # initial guess for key strikes = forward
        initial_guess_with_strikes = [self.forward] * 4 + initial_guess

        def eq_atm(fit_params):
            # check if atm vol quote is equal to implied vol at atm strike
            implied_vol_atm = self.implied_vol(strike=self.k_atm, params=fit_params)
            residual = self.vol_atm - implied_vol_atm

            return residual

        def eq_rr(k_rr_call, k_rr_put, fit_params):
            # check if risk reversal vol quote is satisfied
            vol_rr_call = self.implied_vol(strike=k_rr_call, params=fit_params)
            vol_rr_put = self.implied_vol(strike=k_rr_put, params=fit_params)
            residual = self.vol_rr - (vol_rr_call - vol_rr_put)

            return residual

        def eq_bf(k_bf_call, k_bf_put, fit_params):
            # check if bf vol quote is satisfied
            vol_bf_call = self.implied_vol(strike=k_bf_call, params=fit_params)
            vol_bf_put = self.implied_vol(strike=k_bf_put, params=fit_params)

            price_model = self.option_price(strike=k_bf_call, vol=vol_bf_call, option_type="c") + \
                self.option_price(strike=k_bf_put, vol=vol_bf_put, option_type="p")

            price_bf = self.option_price(strike=k_bf_call, vol=self.vol_atm + self.vol_bf, option_type="c") + \
                self.option_price(strike=k_bf_put, vol=self.vol_atm + self.vol_bf, option_type="p")

            residual = price_bf - price_model

            return residual

        def _delta_diff(strike, fit_params, option_type, desired_delta, vol=None):
            if vol is None:
                vol = self.implied_vol(strike=strike, params=fit_params)
            return self.delta(strike=strike, vol=vol, option_type=option_type) - desired_delta

        def eq_rr_call(k_rr_call, fit_params):
            # check if delta of rr call is satisfied
            return _delta_diff(k_rr_call, fit_params, "c", self.rr_delta)

        def eq_rr_put(k_rr_put, fit_params):
            # check if delta of rr put is satisfied
            return _delta_diff(k_rr_put, fit_params, "p", -self.rr_delta)

        def eq_bf_call(k_bf_call, fit_params):
            # check if delta of bf call is satisfied
            return _delta_diff(k_bf_call, fit_params, "c", self.bf_delta, vol=self.vol_atm + self.vol_bf)

        def eq_bf_put(k_bf_put, fit_params):
            # check if delta of bf put is satisfied
            return _delta_diff(k_bf_put, fit_params, "p", -self.bf_delta, vol=self.vol_atm + self.vol_bf)

        def fit_residuals(fit_params_with_strikes):
            # wrapper for the equations

            # make strikes positive
            k_rr_call, k_rr_put, k_bf_call, k_bf_put = np.abs(fit_params_with_strikes[:4])
            # map params to allowable range of values
            fit_params = self._transform_params_func(fit_params_with_strikes[4:])

            return [eq_atm(fit_params),
                    eq_rr(k_rr_call, k_rr_put, fit_params),
                    eq_bf(k_bf_call, k_bf_put, fit_params),
                    eq_rr_call(k_rr_call, fit_params),
                    eq_rr_put(k_rr_put, fit_params),
                    eq_bf_call(k_bf_call, fit_params),
                    eq_bf_put(k_bf_put, fit_params)]

        solution = optimize.least_squares(fun=fit_residuals, x0=np.array(initial_guess_with_strikes),
                                          ftol=self.tol, xtol=self.tol, gtol=self.tol)

        self.fitted = solution.success
        if self.fitted:
            print("Fit successful.")

            answers = solution.x.tolist()
            self.fitted_params = self._transform_params_func(answers[4:])
            self.k_rr_call, self.k_rr_put, self.k_bf_call, self.k_bf_put = np.abs(answers[:4])

            vols = [self.implied_vol(strike=strike) for strike in answers[:4]]
            self.vol_rr_call, self.vol_rr_put, self.vol_bf_call, self.vol_bf_put = vols

            strike_names = ["k_atm", "k_rr_call", "k_rr_put", "k_bf_call", "k_bf_put"]
            vol_names = ["vol_atm", "vol_rr_call", "vol_rr_put", "vol_bf_call", "vol_bf_put"]
            self.strikes_dict = {strike_name: strike for strike_name, strike
                                 in zip(strike_names, [self.k_atm] + list(np.abs(answers[:4])))}
            self.vols_dict = {vol_name: vol for vol_name, vol in zip(vol_names, [self.vol_atm] + vols)}
        else:
            print("Fit failed. " + solution.message)

        return solution

    def _check_results(self, k_atm=None, vol_atm=None,
                       k_rr_call=None, k_rr_put=None, vol_rr_call=None, vol_rr_put=None,
                       k_bf_call=None, k_bf_put=None, vol_bf_call=None, vol_bf_put=None,
                       fitted_params=None, tol=None):

        inputs_given = [k_atm, vol_atm,
                        k_rr_call, k_rr_put, vol_rr_call, vol_rr_put,
                        k_bf_call, k_bf_put, vol_bf_call, vol_bf_put,
                        fitted_params, tol]

        if None in inputs_given[:-1]:
            assert self.fitted, "Input missing"

        fit_results = [self.k_atm, self.vol_atm,
                       self.k_rr_call, self.k_rr_put, self.vol_rr_call, self.vol_rr_put,
                       self.k_bf_call, self.k_bf_put, self.vol_bf_call, self.vol_bf_put,
                       self.fitted_params, self.tol]

        inputs_given = [inputs_given[i] if inputs_given[i] is not None
                        else fit_results[i]
                        for i in range(len(inputs_given))]

        k_atm, vol_atm, k_rr_call, k_rr_put, vol_rr_call, vol_rr_put, \
            k_bf_call, k_bf_put, vol_bf_call, vol_bf_put, \
            fitted_params, tol = inputs_given

        def success_check(true, pred, tolerance=tol):
            return str(np.abs(true - pred) <= tolerance)

        # Check atm
        print("ATM conditions")
        print("required vol_atm: {:.5f}".format(self.vol_atm))
        print("k_atm given: {:.5f}".format(k_atm))
        implied_atm = self.implied_vol(k_atm, fitted_params)
        print("vol(k_atm) from curve: {:.5f}".format(implied_atm))
        print("atm condition: " + success_check(self.vol_atm, implied_atm))
        print("")

        # Check rr
        print("RR conditions")
        print("required vol_rr: {:.5f}".format(self.vol_rr))
        print("k_rr_call given: {:5f}".format(k_rr_call))
        print("vol_rr_call given: {:.5f}".format(vol_rr_call))
        delta_rr_call = self.delta(k_rr_call, vol_rr_call, "c")
        print("delta(k_rr_call, vol_rr_call): {:.5f}".format(delta_rr_call))
        print("delta_rr_call condition: " + success_check(delta_rr_call, self.rr_delta))
        delta_rr_put = self.delta(k_rr_put, vol_rr_put, "p")
        print("k_rr_put given: {:5f}".format(k_rr_put))
        print("vol_rr_put given: {:.5f}".format(vol_rr_put))
        print("delta(k_rr_put, vol_rr_put): {:.5f}".format(delta_rr_put))
        print("delta_rr_put condition: " + success_check(delta_rr_put, -self.rr_delta))
        vol_diff = vol_rr_call - vol_rr_put
        print("vol_rr_call - vol_rr_put: {:.5f}".format(vol_diff))
        print("vol_rr condition: " + success_check(vol_diff, self.vol_rr))
        print("")

        # Check bf
        print("BF conditions")
        print("required vol_bf: {:.5f}".format(self.vol_bf))

        print("k_bf_call given: {:.5f}".format(k_bf_call))
        print("vol_bf_call given: {:.5f}".format(vol_bf_call))
        delta_bf_call = self.delta(k_bf_call, self.vol_atm + self.vol_bf, "c")
        print("delta(k_bf_call, vol_atm + vol_bf): {:.5f}".format(delta_bf_call))
        print("delta_bf_call condition: " + success_check(delta_bf_call, self.bf_delta))

        print("k_bf_put given: {:.5f}".format(k_bf_put))
        print("vol_bf_put given: {:.5f}".format(vol_bf_put))
        delta_bf_put = self.delta(k_bf_put, self.vol_atm + self.vol_bf, "p")
        print("delta(k_bf_put, vol_atm + vol_bf): {:.5f}".format(delta_bf_put))
        print("delta_bf_put condition: " + success_check(delta_bf_put, -self.bf_delta))

        price_model = self.option_price(k_bf_call, vol_bf_call, "c") + \
            self.option_price(k_bf_put, vol_bf_put, "p")
        price_bs = self.option_price(k_bf_call, self.vol_atm + self.vol_bf, "c") + \
            self.option_price(k_bf_put, self.vol_atm + self.vol_bf, "p")
        print("call_price(k_bf_call, vol_bf_call) + put_price(k_bf_put, vol_bf_put): {:.5f}".format(price_model))
        print("call_price(k_bf_call, vol_atm + vol_bf) + put_price(k_bf_put, vol_atm + vol_bf): " +
              "{:.5f}".format(price_bs))
        print("price_bf condition: " + success_check(price_model, price_bs))
