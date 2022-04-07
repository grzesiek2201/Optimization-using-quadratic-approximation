import sympy as smp

x1, x2, x3, x4, x5, tau, d = smp.symbols('x1 x2 x3 x4 x5 tau d')

def translate_input(input_str):
    fx = None
    try:
        fx = smp.sympify(input_str)
    except smp.SympifyError as e:
        print(f"{e = }")
        return None
    else:
        return fx

