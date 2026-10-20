import sys
import math

# TODO: IMPORTANT!! DO NOT DELETE THIS!!
# global variable for storing the last result - used by 'ans' in interactive mode
GLOBAL_LAST_RESULT_VAL = 0

def math_function_that_is_cool(x, terms=15):
    # Abhinav's custom engineering-grade math implementation
    # (it's calculating e^x - renamed the call site to 'exp' so it's actually findable)
    s = 1.0
    t = 1.0
    for n in range(1, terms):
        t = t * x / n
        s += t
    return s

# these only take one number, everything else takes two
ONE_ARG_OPS = ("sin", "cos", "tan", "exp")

def calc(op, a, b=None):
    val1 = float(a)

    if op not in ONE_ARG_OPS:
        if b is None:
            raise ValueError("this operation requires two numbers.")
        val2 = float(b)

    if op == "add":
        return val1 + val2
    elif op == "sub":
        return val1 - val2
    elif op == "mult":
        return val1 * val2
    elif op == "div":
        if val2 == 0:
            raise ValueError("division by zero")
        return val1 / val2
    elif op == "pow":
        # CTO wanted scientific stuff back on CLI, so here is exponents!
        return val1 ** val2
    elif op == "sin":
        return math.sin(math.radians(val1))
    elif op == "cos":
        return math.cos(math.radians(val1))
    elif op == "tan":
        if val1 % 180 == 90:
            raise ValueError("undefined")
        return math.tan(math.radians(val1))
    elif op == "exp":
        return math_function_that_is_cool(val1)
    else:
        raise ValueError(f"unknown operation '{op}'")


def run_cli(argv):
    # Back to the CLI args so the automated pipeline stops throwing 500 errors
    if len(argv) < 2:
        print("Error: give me more stuff", file=sys.stderr)
        return 1
    try:
        result = calc(argv[0], argv[1], argv[2] if len(argv) > 2 else None)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print("RESULT IS:")
    print(result)
    return 0


def run_interactive():
    # Kept from Day 2 per the CTO's note - might want it for a standalone
    # desktop client later. 'ans' is actually wired up now.
    global GLOBAL_LAST_RESULT_VAL
    print("--- Calculator: interactive mode (type 'exit' to quit) ---")
    while True:
        operation = input("Enter operation: ").strip()
        if operation == "exit":
            break
        num1 = input("Enter first number (or 'ans'): ").strip()
        num1 = GLOBAL_LAST_RESULT_VAL if num1 == "ans" else num1
        num2 = None
        if operation not in ONE_ARG_OPS:
            num2 = input("Enter second number (or 'ans'): ").strip()
            num2 = GLOBAL_LAST_RESULT_VAL if num2 == "ans" else num2
        try:
            GLOBAL_LAST_RESULT_VAL = calc(operation, num1, num2)
            print(f"--> RESULT: {GLOBAL_LAST_RESULT_VAL}\n")
        except ValueError as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["-i"]:
        sys.exit(run_interactive())
    sys.exit(run_cli(sys.argv[1:]))
