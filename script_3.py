import numpy as np
import matplotlib.pyplot as plt


a = np.random.normal(0.5,0.2,(10000,))
b = np.random.normal(0.5,0.2,(10000,))

c = np.minimum(a,b)


plt.hist(a, bins=100, histtype='step', color='g')
plt.hist(b, bins=100, histtype='step', color='b')
plt.hist(c, bins=100, histtype='step', color='r')

plt.axvline(x=np.mean(a), color='g')
plt.axvline(x=np.mean(b), color='b')
plt.axvline(x=np.mean(c), color='r')

plt.show()
