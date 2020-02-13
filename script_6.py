import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp








# The equation is as follows:
# z = 3x^2+3y^2+4xy -11x-3y-119

z = lambda x,y: -3*x**2-3*y**2+4*x*y -11*x-3*y #-119


x = np.linspace(-10,+10,1000)[..., np.newaxis]
y = np.linspace(-20,+20,1000)[np.newaxis, ...]


z = z(x,y)

# z2 = np.log(z+1000000)


fig, ax = plt.subplots()
t = plt.imshow(z)
fig.colorbar(t)
# fig.colorbar(z, orientation='horizontal')
# fig.colorbar(pos, ax=ax1)


P = [[3,-2],
     [ -2,3]]
P = np.array(P, dtype=np.float)*2
q = np.array([11., +3])

A = np.array([[0,0.]])*0
b = np.array([+119.])*0

G = np.array([[1,0],[0,1.],[-1,0],[0,-1.]])
h = np.array([10,20,10,20])




plt.show()



print(solve_qp(P, q, G, h, A, b))