import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x and y values (centered at 0 for a symmetric field)
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)

# Compute the radial distance from the origin
R = np.sqrt(X**2 + Y**2)

# Define a Gaussian-modulated cosine field
Z = np.exp(-0.1*(X**2 + Y**2)) * np.cos(5*R)

# Create a filled contour plot for the 2D field
plt.figure(figsize=(4, 3))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Field value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Field Plot: Gaussian-Modulated Cosine')
plt.savefig('images/plot-gaussian-modulated-cosine-field.png')
plt.close()