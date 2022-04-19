from sympy import Poly
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt

x1, x2, x3, x4, x5, tau, d = smp.symbols('x1 x2 x3 x4 x5 tau d')


def substitute(func, xes=(x1, x2), values=(1, 1)):
    for x, val in zip(xes, values):
        func = func.subs(x, val)
    # func = func.subs(x1, values[0])
    # # print(f"substitute {values[0] = }")
    # func = func.subs(x2, values[1])
    # func = func.subs(x3, values[2])
    # func = func.subs(x4, values[3])
    # func = func.subs(x5, values[4])
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
    return ((func(a)*(tau-b)*(tau-c)) / ((a-b)*(a-c)) + (func(b)*(tau-a)*(tau-c)) / ((b-a)*(b-c)) +
            (func(c)*(tau-a)*(tau-b)) / ((c-a)*(c-b)))


def quadratic_minimum(func):
    polynomial = Poly(func)
    coeffs = polynomial.all_coeffs()
    # 3.03282947879229e+15*(tau - 3.47826086956522)*(12.9384615384616*tau - 41.8011834319529) - 3.03282947879231e+15*(tau - 3.47826086956522)*(12.9384615384616*tau - 41.8011834319529) + 16.3259678597517*(tau - 3.23076923076923)*(13.296786389414*tau - 42.95884833503)???
    # print(f"{func = }, {polynomial = }, {coeffs = }")
    # if polynomial.degree > 1:
    delta = coeffs[1]**2 - 4*coeffs[0]*coeffs[2]
    minimum_y = -delta / (4*coeffs[0])
    if coeffs[0] == coeffs[1]:
        minimum_x = coeffs[0]
    else:
        minimum_x = -coeffs[1] / (2*coeffs[0])
    return float(minimum_x), float(minimum_y)


def algorithm(func, a=-1, c=1, d=(1, 0), x0=(0, 0), epsilon1=0.001, epsilon2=0.001, num_of_iterations=20):
    """ Runs the algorithm.

        Parameters:
            func(sympy lambdified): The

        Returns:
            data: {
                    "steps_x": consequent x values through the steps,
                    "optimized_x": value of the last point, which is also contained in steps_x,
                    "iterations": number of iterations,
                    "x": x
                }

    """
    tau_n = None
    b = (a + c) / 2
    c_max = c
    a_min = a
    iteration = 0
    consequent_x = [[x] for x in list(x0)]  # [x0[0]], [x0[1]]]
    # cond_1 = abs(c - a)
    # cond_3 = abs(func(a) - func(c))
    break_condition = None
    while True:  # 3 conditions for the while loop
        if iteration >= num_of_iterations:
            break_condition = 2
            break
        # print(f"{Poly(func_org) = }, {a = }, {b = }, {c = }")
        approx = quadr_approx(func, a, b, c)
        # print(Poly(approx))

        # #testing
        print(f"iteration: {iteration}, a={a}, b={b}, c={c}")
        # t = np.linspace(-25, 25, 1000)
        # plt.figure()
        # plt.plot(t, func(t))
        # plt.scatter([a, b, c], [func(a), func(b), func(c)])
        # plt.show()
        # #testing

        try:
            tau_n, _ = quadratic_minimum(approx)
        except Exception as e:
            print("***********")
            print(e)
            print("***********")
        print(f"{tau_n = }")

        # if tau_n is not in [a, c] then end the algorithm, the optimum a or c for which func() has the lowest value?

        # if tau_n is at a or c
        if tau_n == a or tau_n == c: ######### not well tested yet
            break

        # if tau_n is outside the specified range,
        # return the value of which the function takes the smaller value
        # x_max = x0 + w * d  ###############??????????????????
        # x_next = x0 + tau_n * d  ###############??????????????????
        # cond = abs(x_next) - abs(x_max)  ###############??????????????????
        if tau_n < a:   # was a_min
            tau_n = a
            break
        if tau_n > c:   # was c_max
            tau_n = c
            break
            # or tau_n > c_max:# or abs(x_next) - abs(x_max) > 0:# or tau_n < a or tau_n > c:
            # print(f"tau_n < a or tau_n > c_max, {tau_n =}")
            # a_val = func(a)
            # c_val = func(c)
            # if a_val < c_val:
            #     tau_n = 0
            # else:
            #     tau_n = w
            # break_condition = 3#################
            # break

        for i in range(len(x0)):
            consequent_x[i].append((x0 + tau_n * d)[i])
        # consequent_x[0].append((x0 + tau_n * d)[0])  # was tau_n / 2
        # consequent_x[1].append((x0 + tau_n * d)[1])  # was tau_n / 2

        if b < tau_n:
            if func(b) >= func(tau_n):
                # print(f"{b} < {tau_n} && {func(b)} >= {func(tau_n)}")
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
                c = b
                b = tau_n
            else:
                # print(f"{b} >= {tau_n} && {func(b)} < {func(tau_n)}")
                a = tau_n
                b = b
                c = c
        else:  # tau_n == b, czyli b jest w minimum
            a = b
            c = b
            iteration += 1
            break_condition = 3###################
            break
        iteration += 1
        # print(iteration)

        if abs(c-a) < epsilon1:
            break_condition = 1
            break
        if abs(func(a) - func(c)) < epsilon2:
            break_condition = 3
            break

    print(f"******* break condition = {break_condition} ********")

    if tau_n:
        tau = tau_n
    else:
        tau = 0
    print(tau)
    print(f"x0({x0} + tau({tau} * d{d} = optimized_x")
    optimized_x = x0 + tau * d
    print(f"{optimized_x = }")

    for i in range(len(x0)):
        consequent_x[i].append(optimized_x[i])
    # consequent_x[0].append(optimized_x[0])
    # consequent_x[1].append(optimized_x[1])

    data = {
        "steps_x": consequent_x,
        "optimized_x": optimized_x,
        "iterations": iteration,
    }

    return data
