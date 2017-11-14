import matplotlib.pyplot as plt
import numpy as np

x = [1, 3, 5, 7, 9, 11, 13, 15]
y =[
    4.96068859863, 4.04489851615, 3.48132555098,
    3.27662439109, 3.36824181432, 3.39875161823,
    3.41406529665, 3.4468418368
]


plt.figure()
plt.plot(x, y)
plt.xlabel("Iterations")
plt.ylabel("Classification of accuracy (%)")
plt.show()
