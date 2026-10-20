# Calculator

```
python calculator.py add 2 3        # prints "RESULT IS:" then 5.0
python calculator.py exp 1          # single-number ops only need one number
python calculator.py -i             # interactive mode, 'ans' reuses the last result
python gui.py                       # Tk window, needs tkinter (not required)
```

Operations: `add sub mult div pow` (two numbers), `sin cos tan exp` (one number, trig in degrees).  
Errors go to stderr and the exit code is non-zero (0 on success, 2 on bad usage, 1 otherwise).