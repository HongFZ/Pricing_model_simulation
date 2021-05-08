import numpy as np
from utilis import Price, HestonPrice, payoff_call
import matplotlib.pyplot as plt


def survive_barrier(St, step, U, L):
    if step <= 1/2:
        if St > U:
            return False
    elif 8/12 <= step <= 11/12:
        if St < L:
            return False
    elif 11/12 <= step <= 1:
        if St < 1.3 or St > 1.8:
            return False
    return True


def simulation(N, M, T, S0, U, L, mu, sigma, X, monitor_len=1):
    step_size = T / N
    St = S0
    steps = [step_size*i for i in range(1, N+1)]
    payoff_seq, price_process_seq, monitor_step = [], [], []

    for _ in range(M):
        price_process = [S0]
        S = Price(mu, sigma, step_size, S0)
        survive = True
        for i, step in enumerate(steps):
            if survive:
                St = S.price_next_step()
                price_process.append(St)
                if (i+1) % monitor_len == 0:
                    survive = survive_barrier(St, step, U, L)
                    monitor_step.append(step)
            else:
                break
        if survive:
            payoff_seq.append(payoff_call(St, X))
            price_process_seq.append(price_process)
        else:
            payoff_seq.append(0)

    return np.mean(payoff_seq), np.std(payoff_seq), payoff_seq, price_process_seq, [0]+steps, monitor_step


def simualtion_Heston_Model(N, M, T, S0, V0, U, L, X, kappa, theta, epsilon, rho, monitor_len=1):
    dt = T / N
    St = S0
    steps = [dt * i for i in range(1, N + 1)]
    payoff_seq, price_process_seq, monitor_step = [], [], []

    for _ in range(M):
        price_process = [S0]
        S = HestonPrice(kappa, theta, epsilon, rho, S0, V0, dt)
        survive = True
        for i, step in enumerate(steps):
            if survive:
                St = S.price_next_step()
                price_process.append(St)
                if (i + 1) % monitor_len == 0:
                    survive = survive_barrier(St, step, U, L)
                    monitor_step.append(step)
            else:
                break
        if survive:
            payoff_seq.append(payoff_call(St, X))
            price_process_seq.append(price_process)
        else:
            payoff_seq.append(0)

    return np.mean(payoff_seq), np.std(payoff_seq), payoff_seq, price_process_seq, [0]+steps, monitor_step


if __name__ == '__main__':
    N = 1000    # number of steps for each paths
    M = 10000   # number of simulation paths
    T = 1       # maturity time, one year
    S0 = 1.5    # initial price of underlying asset
    U = 1.71    # upper bound of barrier, 1-6 month
    L = 1.29    # lower bound of barrier, 8-11 month
    X = 1.4     # strike price

    ## Pricing using constant volatility
    mu = 0.03
    sigma = 0.12

    # continue monitoring
    mean_payoff, std_payoff, payoff_seq, price_processs, steps, _ = simulation(N, M, T, S0, U, L, mu, sigma, X)

    plt.figure(figsize=(12, 8))
    for i in price_processs[:10]:
        plt.plot(steps, i)
    plt.hlines(U, 0, 1 / 2, colors='r', linestyles='dotted')
    plt.hlines(L, 8 / 12, 11 / 12, colors='r', linestyles='dotted')
    plt.hlines(1.3, 11 / 12, 1, colors='r', linestyles='dotted')
    plt.hlines(1.8, 11 / 12, 1, colors='r', linestyles='dotted')
    plt.xlabel('t')
    plt.ylabel('St')
    plt.title('Barrier Option Path Simulation')
    plt.show()

    # discrete monitoring
    mean_payoff, std_payoff, payoff_seq, price_processs, steps, monitor_step = simulation(N, M, T, S0, U, L, mu, sigma, X, monitor_len=50)

    plt.figure(figsize=(12, 8))
    for i in price_processs[:10]:
        plt.plot(steps, i)
    plt.hlines(U, 0, 1 / 2, colors='r', linestyles='dotted')
    plt.hlines(L, 8 / 12, 11 / 12, colors='r', linestyles='dotted')
    plt.hlines(1.3, 11 / 12, 1, colors='r', linestyles='dotted')
    plt.hlines(1.8, 11 / 12, 1, colors='r', linestyles='dotted')
    for i in monitor_step:
        plt.vlines(i, 1, 2, colors='b', linestyles='dashed')
    plt.xlabel('t')
    plt.ylabel('St')
    plt.title('Barrier Option Path Simulation')
    plt.show()


    ## Pricing using Heston Stochastic Volatility Model
    V0 = 0.01       # initial variance
    kappa = 2       # speed of reversion
    theta = 0.12    # long run variance
    epsilon = 0.1   # variance's volatility
    rho = 0         # correlation of random errors

    # continue monitoring
    mean_payoff, std_payoff, payoff_seq, price_processs, steps, _ = simualtion_Heston_Model(N, M, T, S0, V0, U, L, X, kappa, theta, epsilon, rho)

    plt.figure(figsize=(12, 8))
    for i in price_processs[:10]:
        plt.plot(steps, i)
    plt.hlines(U, 0, 1 / 2, colors='r', linestyles='dotted')
    plt.hlines(L, 8 / 12, 11 / 12, colors='r', linestyles='dotted')
    plt.hlines(1.3, 11 / 12, 1, colors='r', linestyles='dotted')
    plt.hlines(1.8, 11 / 12, 1, colors='r', linestyles='dotted')
    plt.xlabel('t')
    plt.ylabel('St')
    plt.title('Barrier Option Path Simulation, Heston Stochastic Volatility Model')
    plt.show()

    # discrete monitoring
    mean_payoff, std_payoff, payoff_seq, price_processs, steps, monitor_step = simualtion_Heston_Model(N, M, T, S0, V0, U, L, X,
                                                                                            kappa, theta, epsilon, rho)


    ## Replicate 10 times
    model_type = 2
    iters = 10
    mean_payoff, std_payoff = [], []
    for _ in range(iters):
        if model_type == 1:
            mean_value, std_value, _, _, _, _ = simulation(N, M, T, S0, U, L, mu, sigma, X)
        elif model_type == 2:
            mean_value, std_value, _, _, _, _ = simulation(N, M, T, S0, U, L, mu, sigma, X, monitor_len=2000)
        elif model_type == 3:
            mean_value, std_value, _, _, _, _ = simualtion_Heston_Model(N, M, T, S0, V0, U, L, X, kappa, theta, epsilon, rho)
        else:
            mean_value, std_value, _, _, _, _ = simualtion_Heston_Model(N, M, T, S0, V0, U, L, X, kappa, theta, epsilon,
                                                                        rho, monitor_len=50)
        mean_payoff.append(mean_value)
        std_payoff.append(std_value)

    print('Mean of Payoff: ', mean_payoff)
    print('STD of Payoff: ', std_payoff)
    print('SE of Payoff: ', np.array(std_payoff) / np.sqrt(M))
    print('Estimated Price: ', np.mean(mean_payoff))


    ## price with different monitor span
    monitor_lens = [i for i in range(1, 51)]
    mean_payoff, std_payoff = [], []
    for monitor_len in monitor_lens:
        mean_value, std_value, _, _, _, _ = simulation(N, M, T, S0, U, L, mu, sigma, X, monitor_len=monitor_len)
        mean_payoff.append(mean_value)
        std_payoff.append(std_value)
    plt.plot(monitor_lens, mean_payoff)
    plt.xlabel('monitor span')
    plt.ylabel('St')
    plt.title('Price with different monitor time span')
    plt.show()


