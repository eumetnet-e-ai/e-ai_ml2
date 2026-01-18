import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 120)
y = np.linspace(0, 3, 90)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

plt.imshow(Z, origin="lower",
     extent=[0,4,0,3], cmap="viridis")
plt.colorbar(); plt.title("2d Field"); 
plt.savefig("plot-2d-field.png"); 
plt.close()
