from sympy import Poly
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='progress_log.log', level=logging.INFO)

x1, x2, x3, x4, x5, tau, d = smp.symbols('x1 x2 x3 x4 x5 tau d')


def substitute(func, xes=(x1, x2), values=(1, 1)):
    tau = smp.symbols('tau')
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
    # if a == b:
    #     a -= 0.00001
    # if a == c:
    #     a -= 0.00001
    # if b == c:
    #     c += 0.00001
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
    iteration = 0
    consequent_x = [[x] for x in list(x0)]
    break_condition = None
    while True:  # 3 conditions for the while loop
        if iteration >= num_of_iterations:
            break_condition = 2
            break
        approx = quadr_approx(func, a, b, c)
        if approx == 0:
            print("approx is 0, cannot process")

        # #testing
        print(f"iteration: {iteration}, a={a}, b={b}, c={c}")

        try:
            tau_n, _ = quadratic_minimum(approx)
        except Exception as e:
            print("***********")
            print(e)
            print("***********")
            raise Exception("Wektor zmiennych musi być tego samego wymiaru, co wektor wartości początkowych "
                            "oraz wektor poszukiwań.\n" + str(e))
        print(f"{tau_n = }")

        ####
        logging.info(
            f"a={round(a, 3)}\t\tb={round(b, 3)}\t\tc={round(c, 3)}\t\t{tau_n=}\t\t\t{x0 + tau_n * d=}\t\tbreak_condition={break_condition}")
        ####

        # if tau_n is not in [a, c] then end the algorithm, the optimum a or c for which func() has the lowest value?

        # if tau_n is at a or c
        if tau_n == a or tau_n == c: ######### not well tested yet
            break_condition = 1
            break

        if tau_n < a:   # was a_min
            break_condition = 1
            tau_n = a
            break
        if tau_n > c:   # was c_max
            break_condition = 1
            tau_n = c
            break

        for i in range(len(x0)):
            consequent_x[i].append((x0 + tau_n * d)[i])

        if b < tau_n:
            if func(b) >= func(tau_n):
                a = b
                b = tau_n
                c = c
            else:
                a = a
                b = b
                c = tau_n
        elif b > tau_n:
            if func(b) >= func(tau_n):
                a = a
                c = b
                b = tau_n
            else:
                a = tau_n
                b = b
                c = c
        else:  # tau_n == b, czyli b jest w minimum
            iteration += 1
            break_condition = 1###################
            break
        iteration += 1

        if abs(c-a) < epsilon1 or abs(c-b) < epsilon1 or abs(b-a) < epsilon1:  # second condition added because of errors when approximating with Poly
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
    print(f"x0({x0}) + tau({tau} * d{d}) = optimized_x")
    optimized_x = x0 + tau * d
    print(f"{optimized_x = }")

    for i in range(len(x0)):
        consequent_x[i].append(optimized_x[i])

    func_at_min = func(tau)

    if break_condition == 1:
        break_condition = "szer. przedziału"
    elif break_condition == 2:
        break_condition = "liczba iteracji"
    elif break_condition == 3:
        break_condition = "wartość funkcji"

    data = {
        "steps_x": consequent_x,
        "optimized_x": optimized_x,
        "iterations": iteration,
        "end": break_condition,
        "func_val_at_min": func_at_min
    }

    return data


def algorithm_step(func, a=-1, c=1, d=(1, 0), x0=(0, 0), epsilon1=0.001, epsilon2=0.001, num_of_iterations=20,
                   iteration=0, consequent_x=None, b=0, prev_tau=0, prev_approx_func=None, prev_optimized_x=None):
    """ Runs one step of the algorithm.

        Parameters:
            func(sympy lambdified): The

        Returns:
                algorithm_data = {
                    "steps_x": consequent_x,
                    "optimized_x": optimized_x,
                    "iterations": iteration,
                    "range": (a, c),
                    "func": func,
                    "x0": x0,
                    "d": d,
                    "l": num_of_iterations,
                    "e1": epsilon1,
                    "e2": epsilon2,
                    "end": break_condition,
                    "b": b
    }

    """
    # data from previous iteration
    prev_algorithm_data = {
        "steps_x": consequent_x,
        "optimized_x": prev_optimized_x,
        "iterations": iteration,
        "range": (a, c),
        "func": func,
        "x0": x0,
        "d": d,
        "l": num_of_iterations,
        "e1": epsilon1,
        "e2": epsilon2,
        "end": None,
        "b": b,
        "tau": prev_tau,
        "approx_func": prev_approx_func,
        "org_func": func,
    }

    tau_n = None
    b = b
    iteration = iteration
    break_condition = None

    approx = None

    for _ in range(1):
        # 3 conditions for the while loop
        if iteration >= num_of_iterations:
            break_condition = 2
            prev_algorithm_data["end"] = break_condition
            return prev_algorithm_data

        approx = quadr_approx(func, a, b, c)

        # #testing
        print(f"iteration: {iteration}, a={a}, b={b}, c={c}")

        try:
            tau_n, _ = quadratic_minimum(approx)
        except Exception as e:
            print("***********")
            print(e)
            print("***********")
            raise Exception("Wektor zmiennych musi być tego samego wymiaru, co wektor wartości początkowych "
                            "oraz wektor poszukiwań.\n" + str(e))
        print(f"{tau_n = }")

        # if tau_n is not in [a, c] then end the algorithm, the optimum a or c for which func() has the lowest value?

        # # if tau_n is at a or c
        # if tau_n == a or tau_n == c: ######### not well tested yet
        #     break_condition = 1
        #     break

        if tau_n <= a:   # was a_min
            break_condition = 1  # should it be 1?
            tau_n = a
            break
        if tau_n >= c:   # was c_max
            break_condition = 1  # should it be 1?
            tau_n = c
            break

        for i in range(len(x0)):
            consequent_x[i].append((x0 + tau_n * d)[i])

        if b < tau_n:
            if func(b) >= func(tau_n):
                a = b
                b = tau_n
                c = c
            else:
                a = a
                b = b
                c = tau_n
        elif b > tau_n:
            if func(b) >= func(tau_n):
                a = a
                c = b
                b = tau_n
            else:
                a = tau_n
                b = b
                c = c
        else:  # tau_n == b, czyli b jest w minimum
            iteration += 1
            break_condition = 1
            break
        iteration += 1

        if abs(c-a) < epsilon1 or abs(c-b) < epsilon1 or abs(b-a) < epsilon1:  # second condition added because of errors when approximating with Poly
            break_condition = 1
            break
        if abs(func(a) - func(c)) < epsilon2:
            break_condition = 3
            break

    print(f"******* break condition = {break_condition} ********")

    ####
    try:
        logging.info(f"a={round(a, 3)}\t\tb={round(b, 3)}\t\tc={round(c, 3)}\t\t{tau_n=}\t\t\t{x0 + tau_n * d=}\t\tbreak_condition={break_condition}")
    except Exception:
        pass
    ####

    if tau_n:
        tau = tau_n
    else:
        tau = 0
    print(tau)
    print(f"x0({x0}) + tau({tau} * d{d}) = optimized_x")
    optimized_x = x0 + tau * d
    print(f"{optimized_x = }")

    for i in range(len(x0)):
        consequent_x[i].append(optimized_x[i])

    func_at_min = func(tau)

    tau_symbol = smp.symbols('tau')
    approx_func = smp.lambdify(tau_symbol, approx, 'numpy')

    if break_condition == 1:
        break_condition = "szer. przedziału"
    elif break_condition == 2:
        break_condition = "liczba iteracji"
    elif break_condition == 3:
        break_condition = "wartość funkcji"

    algorithm_data = {
        "steps_x": consequent_x,
        "optimized_x": optimized_x,
        "iterations": iteration,
        "range": (a, c),
        "func": func,
        "x0": x0,
        "d": d,
        "l": num_of_iterations,
        "e1": epsilon1,
        "e2": epsilon2,
        "end": break_condition,
        "b": b,
        "tau": tau,
        "approx_func": approx_func,
        "org_func": func,
        "func_val_at_min": func_at_min
    }

    return algorithm_data

# break_condition = 1 - epsilon 1 (proximity of the section)
# break_condition = 2 - iterations
# break_condition = 3 - epsilon 2 (proximity of the values of the function at the ends of section)
