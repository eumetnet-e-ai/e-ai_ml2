import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(4,3))
plt.plot(x, y)
plt.xlabel("x"); plt.ylabel("sin(x)")
plt.title('Sine Wave')                                                                      
plt.tight_layout()
plt.savefig("plot-sine-wave.png")
plt.close()
