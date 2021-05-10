import numpy as np
import matplotlib.pyplot as plt
from utilis import EuropeanOption, GeometricBrownianMotion, payoff_call


if __name__ == '__main__':
    strike = 430
    risk_free_rate = .01
    initial_price = 400
    drift = .03
    volatility = .12
    T = 1

    dt = T / 1000
    paths_num = [i for i in range(1, 5050, 5)]
    paths = 10000
    pricing_mean, pricing_std = [], []

    for s in range(100, 1000, 10):
        dt = T / s
        price_paths, call_payoffs = [], []
        # Generate a set of sample paths
        for i in range(0, paths):
            price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)
        # calculate payoff for each path
        for price_path in price_paths:
            p = payoff_call(price_path[-1], strike)
            call_payoffs.append(p*np.exp(-risk_free_rate*T))
        pricing_mean.append(np.mean(call_payoffs))
        pricing_std.append(np.std(call_payoffs))

    # Plot the set of generated sample paths
    # for price_path in price_paths[:10]:
    #     plt.plot(price_path)
    # plt.title('Price Path Simulation')
    # plt.show()

    plt.plot([s for s in range(100, 1000, 10)], pricing_std)
    plt.title('Price std with different step number')
    plt.xlabel('step number')
    plt.ylabel('price')
    plt.show()

    # Pricing using Monte Carlo simulation
    print('Price using Monte Carlo simulation (mean): ', pricing_mean)
    print('Price using Monte Carlo simulation (STD): ', pricing_std)
    print('Price using Monte Carlo simulation (SE): ', pricing_std / np.sqrt(len(call_payoffs)))
    print('Price using Monte Carlo simulation: ', np.mean(pricing_mean))

    # Pricing using BS model
    bs_price = EuropeanOption(initial_price, volatility, strike, T, risk_free_rate)
    print('Price using Black Scholes Model: ', bs_price.price_call)

