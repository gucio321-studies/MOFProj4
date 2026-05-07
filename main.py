#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

class Poison:
    def __init__(self,N=31, x0=4, d=4, dx=1, ex2=False):
        self.x0 = x0
        self.d = d
        self.dx = dx
        self.N = N
        self._u = np.zeros((self.N*2+1, self.N*2+1))
        self._a = []
        self._rho_dot = np.zeros((self.N*2+1, self.N*2+1))
        self._delta = np.zeros((self.N*2+1, self.N*2+1))
        self.u_calc = self.u_ex1 if not ex2 else self.u_ex2

    def rho(self, x, y):
        d = self.d
        x0 = self.x0
        return np.exp(-((x-x0)**2+y**2)/(d**2)) - np.exp(-((x+x0)**2+y**2)/(d**2))
    def u(self, i, j):
        return self._u[self.N+i][self.N+j]

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
                self._u[self.N+i][self.N+j] = self.u_calc(self, i, j)
        self._a.append(self.a())
    def iterate_until(self, max_iter=500):
        for _ in range(max_iter - len(self._a)): self.iterate()
    def rho_dot(self, i, j):
        return -1*(self.u(i+1, j) + self.u(i-1, j) + self.u(i, j-1) + self.u(i, j+1) - 4*self.u(i, j)) / self.dx**2
    def gen_rho_dot(self):
        for i in range(-self.N+1, self.N):
            for j in range(-self.N+1, self.N):
                self._rho_dot[self.N+i][self.N+j] = self.rho_dot(i,j)
    def delta(self, i, j):
        return self._rho_dot[self.N+i][self.N+j] - self.rho(i, j)
    def gen_delta(self):
        self.gen_rho_dot()
        for i in range(-self.N+1, self.N):
            for j in range(-self.N+1, self.N):
                self._delta[self.N+i][self.N+j] = self.delta(i, j)

def ex1():
    p = Poison()
    p.iterate_until(100)
    u100 = p._u.copy() # _u is pointer so need to use copy here
    p.gen_delta()
    rho_dot100 = p._rho_dot.copy()
    delta100 = p._delta.copy()

    p.iterate_until(500)
    u500 = p._u.copy()
    p.gen_delta()
    rho_dot500 = p._rho_dot.copy()
    delta500 = p._delta.copy()

    plt.plot(p._a)
    plt.xlabel("iteration")
    plt.ylabel("a")
    plt.grid(True)
    plt.show()

    plt.subplot(1,2,1)
    plt.imshow(u100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u after 100th iteration")
    plt.subplot(1,2,2)
    plt.imshow(u500)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u after 500th iteration")
    plt.show()

    plt.subplot(2,2,1)
    plt.imshow(rho_dot100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$\\rho$' after 100th iteration")
    plt.subplot(2,2,2)
    plt.imshow(rho_dot500)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$\\rho$' after 500th iteration")

    plt.subplot(2,2,3)
    plt.imshow(delta100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$\\delta$ after 100th iteration")
    plt.subplot(2,2,4)
    plt.imshow(delta500)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$\\delta$ after 500th iteration")
    plt.show()


ex1()