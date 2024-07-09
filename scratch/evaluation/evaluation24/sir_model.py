import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import odeint


# simulate SIR model - normalized (populations S,I,R add to 1)
def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -S * (beta(t) * I)
    dIdt = S * (beta(t) * I) - gamma(t) * I
    dRdt = gamma(t) * I

    return dSdt, dIdt, dRdt


#  plot SIR model
def plotSIR(t, S, I, R):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, S, "b", alpha=0.7, linewidth=2, label="Susceptible")
    ax.plot(t, I, "r", alpha=0.7, linewidth=2, label="Infected")
    ax.plot(t, R, "r--", alpha=0.7, linewidth=2, label="Recovered")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fraction of population")

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()


# simulate SIDARTHE model - normalized (populations S,I,R add to 1)
def SIDARTHE_model(
    y,
    t,
    alpha,
    beta,
    gamma,
    delta,
    epsilon,
    mu,
    zeta,
    lamb,
    eta,
    rho,
    theta,
    kappa,
    nu,
    xi,
    sigma,
    tau,
):
    S, I, D, A, R, T, H, E = y
    dSdt = -S * (alpha(t) * I + beta(t) * D + gamma(t) * A + delta(t) * R)
    dIdt = (
        S * (alpha(t) * I + beta(t) * D + gamma(t) * A + delta(t) * R)
        - (epsilon(t) + zeta(t) + lamb(t)) * I
    )
    dDdt = epsilon(t) * I - (eta(t) + rho(t)) * D
    dAdt = zeta(t) * I - (theta(t) + mu(t) + kappa(t)) * A
    dRdt = eta(t) * D + theta(t) * A - (nu(t) + xi(t)) * R
    dTdt = mu(t) * A + nu(t) * R - (sigma(t) + tau(t)) * T
    dHdt = lamb(t) * I + rho(t) * D + kappa(t) * A + xi(t) * R + sigma(t) * T
    dEdt = tau(t) * T

    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


#  plot SIDARTHE model
def plotSIDARTHE(t, S, I, D, A, R, T, H, E):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, S, "b", alpha=0.7, linewidth=2, label="Susceptible")
    ax.plot(
        t,
        I,
        "r",
        alpha=0.7,
        linewidth=2,
        label="Infected (Asymptomatic, Infected, Undetected)",
    )
    ax.plot(
        t,
        D,
        "r.",
        alpha=0.7,
        linewidth=2,
        label="Diagnosed (Asymptomatic, Infected, Detected)",
    )
    ax.plot(
        t,
        A,
        "r:",
        alpha=0.7,
        linewidth=2,
        label="Ailing (Symptomatic, Infected, Undetected)",
    )
    ax.plot(
        t,
        R,
        "r--",
        alpha=0.7,
        linewidth=2,
        label="Recognized (Symptomatic, Infected, Detected)",
    )
    ax.plot(
        t,
        T,
        "r-.",
        alpha=0.7,
        linewidth=2,
        label="Threatened (Acutely Symptomatic)",
    )
    ax.plot(t, H, "g", alpha=0.7, linewidth=2, label="Healed")
    ax.plot(t, E, "k", alpha=0.7, linewidth=2, label="Extinct (Dead)")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fraction of population")

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()


def SIHDR_model(y, t, beta, ae, bf, cg, dh, N):
    S, I, H, D, R = y
    dSdt = -(beta(t) * S * I / N(t))
    dIdt = (beta(t) * S * I / N(t)) - ae(t) * I - bf(t) * I
    dHdt = bf(t) * I - cg(t) * H - dh(t) * H
    dDdt = dh(t) * H
    dRdt = ae(t) * I + cg(t) * H

    return dSdt, dIdt, dHdt, dDdt, dRdt


#  plot SIHDR model
def plotSIHDR(t, S, I, H, D, R):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, S, "b", alpha=0.7, linewidth=2, label="Susceptible")
    ax.plot(t, I, "r", alpha=0.7, linewidth=2, label="Infected")
    ax.plot(t, H, "g", alpha=0.7, linewidth=2, label="Hospitalized")
    ax.plot(t, D, "r.", alpha=0.7, linewidth=2, label="Dead")
    ax.plot(t, R, "r--", alpha=0.7, linewidth=2, label="Recovered")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fraction of population")

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()


def plot_two_params(param1, param2, param_choices_true_false):
    xsample = [result[param1] for result in param_choices_true_false]
    ysample = [result[param2] for result in param_choices_true_false]
    colors = [
        "green" if result["assignment"] == "1" else "red"
        for result in param_choices_true_false
    ]
    plt.title("Parameter Value Pairings")
    plt.xlabel(param1)
    plt.ylabel(param2)
    area = 5  # size of marker
    plt.scatter(xsample, ysample, s=area, c=colors, alpha=0.5)
    plt.show()
