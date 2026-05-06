#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

class Poison:
    def __init__(self,N=31, x0=4, d=4, dx=1):
        self.x0 = x0
        self.d = d
        self.dx = dx
        self.N = N
        self._u = np.zeros((self.N*2+1, self.N*2+1))
        self._a = []
        self.u_calc = self.u_ex1

    def rho(self, x, y):
        d = self.d
        x0 = self.x0
        return np.exp(-((x-x0)**2+y**2)/(d**2)) - np.exp(-((x+x0)**2+y**2)/(d**2))
    def u(self, i, j):
        return self._u[i][j]

    @staticmethod
    def u_ex1(self, i, j):
        u0 = self.u(i+1, j) + self.u(i-1, j) + self.u(i, j+1) + self.u(i, j-1)+ self.rho(i, j)*self.dx**2
        return (u0)/4
    def a(self):
        result = 0
        dx2 = self.dx**2
        for i in range(-self.N+1, self.N): # range(-5, 5) will return [-5 ... 4]
            for j in range(-self.N+1, self.N):
                a1 = .5 * self.u(i,j) * (self.u(i+1, j) + self.u(i-1, j)-2*self.u(i, j))
                a1 /= dx2
                a2 = .5 * self.u(i,j) * (self.u(i, j+1) + self.u(i, j-1) - 2*self.u(i, j))
                a2 /= dx2
                a3 = self.rho(i, j)*self.u(i, j)
                result += (a1 + a2 + a3)*dx2
        return result
    def iterate(self):
        for i in range(-self.N+1, self.N):
            for j in range(-self.N+1, self.N):
                self._u[i][j] = self.u_calc(self, i, j)
        self._a.append(self.a())
    def iterate_until(self, max_iter=500):
        for _ in range(max_iter - len(self._a)): self.iterate()

def ex1():
    p = Poison()
    p.iterate_until()
    plt.plot(p._a)
    plt.xlabel("iteration")
    plt.ylabel("a")
    plt.grid(True)
    plt.show()

ex1()