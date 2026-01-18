import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Lorenz system
s, b, r = 10, 8/3, 28
dt, N = 0.01, 10000
xyz = np.zeros((N, 3))
xyz[0] = 1, 1, 1

for i in range(1, N):
    x, y, z = xyz[i-1]
    dx = s*(y - x)
    dy = x*(r - z) - y
    dz = x*y - b*z
    xyz[i] = x + dt*dx, y + dt*dy, z + dt*dz

# KDE plot with colorbar
sns.set(style="white")
plt.figure(figsize=(6, 5))
kde = sns.kdeplot(
    x=xyz[:,0], y=xyz[:,2], 
    fill=True, cmap="viridis", levels=100, thresh=0.02
)
plt.colorbar(kde.collections[0], label="Density")
plt.title("Lorenz Attractor Density (x vs z)")
plt.xlabel("x"); plt.ylabel("z")
plt.tight_layout()
plt.savefig("images/lorenz63-seaborn.png")