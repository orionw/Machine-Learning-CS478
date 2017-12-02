import matplotlib.pyplot as plt
import numpy as np

x = [2, 3, 4, 5, 6, 7]
y = [12.1436882816, 7.13864770399, 6.0364635105, 5.0032361045, 4.04232871447, 3.74392242009]


plt.figure()
plt.plot(x, y)
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()
