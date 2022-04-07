from sympy import Poly
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt

x1, x2, x3, x4, x5, tau, d = smp.symbols('x1 x2 x3 x4 x5 tau d')


def substitute(func, values=(1, 1)):
    func = func.subs(x1, values[0])
    # print(f"substitute {values[0] = }")
    func = func.subs(x2, values[1])
    # print(f"substitute {values[1] = }")
    # print(f"function after substition {func = }")
    return func



def quadr_approx(func, a, b, c):
    if a == b:
        a -= 0.001
    if a == c:
        a -= 0.001
    if b == c:
        c += 0.001
    return ((func(a)*(tau-b)*(tau-c)) / ((a-b)*(a-c)) + (func(b)*(tau-a)*(tau-c)) / ((b-a)*(b-c)) +\
        (func(c)*(tau-a)*(tau-b)) / ((c-a)*(c-b)))


def quadratic_minimum(func):
    polynomial = Poly(func)
    coeffs = polynomial.all_coeffs()
    delta = coeffs[1]**2 - 4*coeffs[0]*coeffs[2]
    minimum_y = -delta / (4*coeffs[0])
    if coeffs[0] == coeffs[1]:
        minimum_x = coeffs[0]
    else:
        minimum_x = -coeffs[1] / (2*coeffs[0])
    # print(f"{coeffs = }")
    return float(minimum_x), float(minimum_y)


def algorithm(func, a=-1, c=1, d=(1, 0), x0=(0, 0), epsilon1=0.001, epsilon2=0.001, num_of_iterations=20):
    b = (a + c) / 2
    iteration = 0
    consequent_x = [[x0[0]], [x0[1]]]
    while abs(c - a) > epsilon1 and iteration < num_of_iterations and abs(func(a) - func(c)) > epsilon2:  # 3 conditions for the while loop
        # print(f"{Poly(func_org) = }, {a = }, {b = }, {c = }")
        approx = quadr_approx(func, a, b, c)
        # print(Poly(approx))
        consequent_x[0].append((x0 + (a + c) / 2 * d)[0])
        consequent_x[1].append((x0 + (a + c) / 2 * d)[1])

        tau_n, _ = quadratic_minimum(approx)
        # print(f"{tau_n = }")

        if b < tau_n:
            if func(b) >= func(tau_n):
                print(f"{b} < {tau_n} && {func(b)} >= {func(tau_n)}")
                a = b
                b = tau_n
                c = c
            else:
                # print(f"{b} < {tau_n} && {func(b)} < {func(tau_n)}")
                a = a
                b = b
                c = tau_n
        elif b > tau_n:
            if func(b) >= func(tau_n):
                # print(f"{b} >= {tau_n} && {func(b)} >= {func(tau_n)}")
                a = a
                b = tau_n
                c = b
            else:
                # print(f"{b} >= {tau_n} && {func(b)} < {func(tau_n)}")
                a = tau_n
                b = b
                c = c
        else:  # tau_n == b, czyli b jest w minimum
            a = b
            c = b
            iteration += 1
            break
        iteration += 1
        # print(iteration)

    tau = 1 / 2 * (a + c)
    print(tau)
    x = x0 + tau * d
    print(f"{x = }")

    last_point = ((x0 + (a+c)/2 * d)[0], (x0 + (a+c)/2 * d)[1])

    data = {
        "steps_x": consequent_x,
        "last_point": last_point,
        "iterations": iteration,
        "x": x
    }

    return data
