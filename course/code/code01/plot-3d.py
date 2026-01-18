import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 80)
y = np.linspace(0, 3, 60)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
plt.savefig("plot-3d.png",transparent=True); 
plt.close()
