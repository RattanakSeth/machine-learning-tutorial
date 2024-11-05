import numpy as np
import matplotlib.pyplot as plt

"""
This code is license to https://loem-ms.github.io/MLinKHMERpage/SGD.html
"""

class SGD:
    def f(x):
        # x^2 - 2*x -3
        return x ** 2 - 2 * x -3

    def g(x):
        return 2*x - 2

    """
        eta: learning rate
        eps: Epsilon
    """
    def sd(f, g, x=0., eta=0.01, eps=1e-4):
        t = 1
        H = []
        while True:
            gx = g(x)
            # push or append an object(dict) to array
            H.append(dict(t=t, x=x, fx=f(x), gx=gx))
            if -eps < gx < eps:
                break
            x -= eta * gx
            t += 1
        return H
    
    def plot(H):
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(1,1,1)
        ax.plot(
            [h['t'] for h in H],
            [h['fx'] for h in H],
            'x-'
            )
        ax.set_xlabel('$t$')
        ax.set_ylabel('$f(x)$')
        ax.grid()