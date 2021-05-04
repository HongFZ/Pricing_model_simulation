import numpy as np
import matplotlib.pyplot as plt
from utilis import EuropeanOption, GeometricBrownianMotion, payoff_call


if __name__ == '__main__':
    paths = 1000
    initial_price = 4
    drift = .03
    volatility = .12
    dt = 1 / 1000
    T = 1
    # my student ID: 2020212430, use last 3 numbers as strike price
    strike = 4.30
    risk_free_rate = .01
    price_paths, call_payoffs = [], []

    # Generate a set of sample paths
    for i in range(0, paths):
        price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)

    # calculate payoff for each path
    for price_path in price_paths:
        p = payoff_call(price_path[-1], strike)
        call_payoffs.append(p*np.exp(-risk_free_rate*T))

    # Plot the set of generated sample paths
    for price_path in price_paths:
        plt.plot(price_path)
    plt.title('Price Path Simulation')
    plt.show()

    # Pricing using Monte Carlo simulation
    print('Price using Monte Carlo simulation: ', np.mean(call_payoffs))

    # Pricing using BS model
    bs_price = EuropeanOption(initial_price, volatility, strike, T, risk_free_rate)
    print('Price using Black Scholes Model: ', bs_price.price_call)

