import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

class Price(object):
    def __init__(self, mu, sigma, time_span, init_price):
        self._mu = mu
        self._sigma = sigma
        self._time_span = time_span
        self.current_price = init_price

    def price_next_step(self):
        zt = np.random.randn()
        mu = self._mu
        sigma = self._sigma
        t = self._time_span
        price = self.current_price
        price_new = price + mu*price*t + sigma*price*zt*np.sqrt(t)
        self.current_price = price_new
        return price_new


class GeometricBrownianMotion(object):
    def __init__(self, initial_price, mu, volatility, dt, T):
        self.current_price = initial_price
        self.initial_price = initial_price
        self._mu = mu
        self._volatility = volatility
        self._dt = dt
        self._T = T
        self.prices = []
        self.simulate_paths()

    def simulate_paths(self):
        time_remain = self._T
        while(time_remain - self._dt > 0):
            dWt = np.random.normal(0, math.sqrt(self._dt))
            dYt = self._mu*self._dt + self._volatility*dWt
            self.current_price += dYt*self.current_price
            self.prices.append(self.current_price)
            time_remain -= self._dt


class EuropeanCall(object):
    def __init__(self, asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate):
        self._asset_price = asset_price
        self._volatility = asset_volatility
        self._strike_price = strike_price
        self._time_to_expiration = time_to_expiration
        self._risk_free_rate = risk_free_rate
        self.price = self.call_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)

    def call_price(self, asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate):
        b = math.exp(-risk_free_rate * time_to_expiration)
        x1 = math.log(asset_price / strike_price) + (risk_free_rate + .5 * (asset_volatility**2)) * time_to_expiration
        x1 = x1 / (asset_volatility * (np.sqrt(time_to_expiration)))
        z1 = norm.cdf(x1)
        z1 = z1 * asset_price
        x2 = x1 - (asset_volatility * (np.sqrt(time_to_expiration)))
        z2 = norm.cdf(x2)
        z2 = b * strike_price * z2
        return z1 - z2


class EuropeanPut(object):
    def __init__(self, asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.price = self.put_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)

    def put_price(self, asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate):
        b = math.exp(-risk_free_rate * time_to_expiration)
        x1 = math.log((b * strike_price) / asset_price) + .5 * (asset_volatility * asset_volatility) * time_to_expiration
        x1 = x1 / (asset_volatility * (time_to_expiration ** .5))
        z1 = norm.cdf(x1)
        z1 = b * strike_price * z1
        x2 = math.log((b * strike_price) / asset_price) - .5 * (asset_volatility * asset_volatility) * time_to_expiration
        x2 = x2 / (asset_volatility * (time_to_expiration ** .5))
        z2 = norm.cdf(x2)
        z2 = asset_price * z2
        return z1 - z2



def payoff_call(St, K):
    return max(0.0, St - K)

def payoff_put(St, K):
    return max(0.0, K - St)




if __name__ == '__main__':
    # test Geometric Brownian Motion generation
    paths = 10
    initial_price = 1.5
    drift = .08
    volatility = .1
    dt = 1 / 365
    T = 1
    price_paths = []

    for i in range(0, paths):
        price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)

    for price_path in price_paths:
        plt.plot(price_path)
    plt.show()

    # test pricing using BS model
    ec = EuropeanCall(100, .3, 100, 1, .01)
    print(ec.price)