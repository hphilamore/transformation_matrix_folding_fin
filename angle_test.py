import numpy as np
from math import cos, sin, radians, pi
import matplotlib.pyplot as plt

r = 1
L0 = 2

lengths = []
thetas = []

for theta in np.linspace(0, pi):

    L = (2 * r**2 - 2 * r**2 * cos(theta))**(1/2)

    dL = 8 - 4 * (L0 - L)

    lengths.append(dL)
    thetas.append(theta)

    

plt.plot(lengths, thetas)
plt.show()

