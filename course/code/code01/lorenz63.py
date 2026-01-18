import numpy as np
import matplotlib.pyplot as plt

# Parameters and initial condition
sigma, beta, rho = 10, 8/3, 28
dt, steps = 0.01, 10000
xyz = np.empty((steps, 3))
xyz[0] = (1, 1, 1)

# Integration using Euler method
for i in range(steps - 1):
    x, y, z = xyz[i]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    xyz[i + 1] = xyz[i] + dt * np.array([dx, dy, dz])

# Plot the result
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(projection='3d')
ax.plot(*xyz.T, lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_facecolor("white")       # plot area (axes background)
plt.savefig('images/lorenz63.png')
plt.show()