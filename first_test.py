import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


print ("test with a simple function")

t1 = np.arange(0.0, 5.0, 0.01)

plt.plot(t1, f(t1), "b .")

plt.show()


#
# this is the target function
#
#






