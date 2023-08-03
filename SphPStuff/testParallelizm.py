import numpy as np


from functools import partial
from itertools import repeat
from multiprocessing import Pool

def func(a, b, c):
    return a + b + c

def main():
    print("Trying out parallelizsm")

    arr = np.arange(10)
    arg2 = np.arange(10)
    arg3 = 1
    with Pool() as pool:
        res = np.array(pool.starmap(func, zip(arr, repeat(arg2), repeat(arg3))))

    print("Result = ")
    print(res)



if __name__ == "__main__":
    main()