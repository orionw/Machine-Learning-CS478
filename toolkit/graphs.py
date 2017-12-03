import matplotlib.pyplot as plt
import numpy as np

x = [2, 3, 4, 5, 6, 7]
y = [
    -0.0704078173973,
    - 0.306264439299,
    - 0.308193570046,
    - 0.314017324746,
    - 0.318553794791,
    - 0.319055916863
    ]

plt.figure()
plt.plot(x, y)
plt.xlabel("K")
plt.ylabel("Silhouette")
plt.show()
