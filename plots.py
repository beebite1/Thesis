import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def bV(x, Dt, V):
    return np.sqrt((2*x - Dt*V)**2 - (Dt*V)**2)/2

def bH(x, Dt, V):
    return np.sqrt((2*x - Dt*(V))**2 - (Dt*(V))**2)/2

def f(x, A, B, V):
    return A*np.exp(-bV(x, 0.2, V)/B)

def g(x, A, B):
    return A*np.exp((-x)/B)

def h(x, A, B, V):
    return A*np.exp(-bH(x, 2, V)/B)*(2*x - 2*V)/(2*bH(x, 2, V))

A = 2.1
B = 0.3
V = 0.01
X = np.linspace(0, 1.5, 151)
print(X)
F = f(X, A, B, V)
G = g(X, A, B)
H = h(X, A, B, V)

plt.figure()
plt.title("Pedestrian repulsion forces")
plt.plot(X, F, color = 'b', label = "Vadere Implementation", linewidth = 3)
plt.plot(X, G, color = 'r', label = "Circular Assumption", linewidth = 3)
plt.plot(X, H, color = 'orange', label = "Helbing Model", linewidth = 3)
plt.xlabel("Distance between two pedestrians [m]")
plt.ylabel("Repulsion force exerted [N]")
plt.ylim(-0.1, 4)
plt.legend()
plt.grid(True)
plt.show()