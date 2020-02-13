import numpy as np
import matplotlib.pyplot as plt





mean = np.array([0,0])
cov = np.array([[1,0],[0,1]])



A = np.random.multivariate_normal(mean, cov, size=(1000))


print(A.shape)


plt.scatter(x=A[:,0], y=A[:,1],s=0.7)
plt.show()


print(np.mean(A, axis=0))




# print()