import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt


class Price(object):
    def __init__(self, mu, sigma, dt, S0):
        self._mu = mu
        self._sigma = sigma
        self._S0 = S0
        self.dt = dt
        self.St = S0

    def price_next_step(self):
        zt = np.random.randn()
        self.St = self.St + self._mu * self.St * self.dt + self._sigma * self.St * zt * np.sqrt(self.dt)
        return self.St

    def simulation_path(self, n_steps, m_paths):
        St = np.ones(m_paths) * self.St
        St_paths = []
        for _ in range(n_steps):
            zt = np.random.normal(size=m_paths)
            St = St + self._mu * St * self.dt + self._sigma * St * zt * np.sqrt(self.dt)
            St_paths.append(St)
        St_paths = np.array(St_paths).T
        return St_paths


class HestonPrice(object):
    def __init__(self, kappa, theta, epsilon, rho, S0, V0, dt):
        self._kappa = kappa
        self._theta = theta
        self._epsilon = epsilon
        self._rho = rho
        self._S0 = S0
        self._V0 = V0
        self.Vt = V0
        self.St = S0
        self.dt = dt

    def price_next_step(self):
        Zs = np.random.randn()
        Zv = self._rho * Zs + np.sqrt(1 - self._rho ** 2) * np.random.randn()
        self.Vt = max(self.Vt, 0.0000001)
        self.St = self.St * np.exp(np.sqrt(self.Vt * self.dt) * Zs - self.Vt * self.dt / 2)
        self.Vt = self.Vt + self._kappa * (self._theta - self.Vt) * self.dt + self._epsilon * np.sqrt(self.Vt * self.dt) * Zv
        return self.St

    def simulation_path(self, n_steps, m_paths):
        Vt = np.ones(m_paths) * self._V0
        St = np.ones(m_paths) * self._S0
        St_paths, Vt_paths = [], []
        for _ in range(0, n_steps):
            Zs = np.random.normal(size=m_paths)
            Zv = self._rho * Zs + np.sqrt(1 - self._rho ** 2) * np.random.normal(size=n_steps)
            Vt = np.maximum(Vt, 0.0000001)
            St = St * np.exp(np.sqrt(Vt * self.dt) * Zs - Vt * self.dt / 2)
            Vt = Vt + self._kappa * (self._theta - Vt) * dt + self._epsilon * np.sqrt(Vt * self.dt) * Zv
            St_paths.append(St)
            Vt_paths.append(Vt)
        St_paths = np.array(St_paths).T
        Vt_paths = np.array(Vt_paths).T
        return St_paths, Vt_paths


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
        while time_remain - self._dt > 0:
            dWt = np.random.normal(0, math.sqrt(self._dt))
            dYt = self._mu*self._dt + self._volatility*dWt
            self.current_price += dYt*self.current_price
            self.prices.append(self.current_price)
            time_remain -= self._dt


class EuropeanOption(object):
    def __init__(self, asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate):
        self._asset_price = asset_price
        self._volatility = asset_volatility
        self._strike_price = strike_price
        self._time_to_expiration = time_to_expiration
        self._risk_free_rate = risk_free_rate
        self.price_call = self.call_price()
        self.price_put = self.put_price()

    def call_price(self):
        b = math.exp(-self._risk_free_rate * self._time_to_expiration)
        x1 = math.log(self._asset_price / self._strike_price) + (self._risk_free_rate + .5 * (self._volatility**2)) * self._time_to_expiration
        x1 = x1 / (self._volatility * (np.sqrt(self._time_to_expiration)))
        z1 = norm.cdf(x1)
        z1 = z1 * self._asset_price
        x2 = x1 - (self._volatility * (np.sqrt(self._time_to_expiration)))
        z2 = norm.cdf(x2)
        z2 = b * self._strike_price * z2
        return z1 - z2

    def put_price(self):
        b = math.exp(-self._risk_free_rate * self._time_to_expiration)
        return self._strike_price*b - self._asset_price + self.price_call


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
    ec = EuropeanOption(100, .3, 100, 1, .01)
    print(ec.price_call)
    print(ec.price_put)