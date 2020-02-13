import numpy as np
import matplotlib.pyplot as plt





f = lambda x: np.cos(x/10)+0.08*x+np.random.normal(0,0.1,size=x.shape)
f_w_n = lambda x: np.cos(x/10)+0.08*x




x = np.linspace(0,100,1000)
y = f(x)


plt.plot(x,y)
plt.show()



gamma = (1./30.)

our_initial_points = []
mean_f_on_initial_points = []
var_f_on_initial_points = []



for i in range(300):
    point = np.random.uniform(0,100)
    points = np.repeat([point], repeats=10, axis=0)
    points_f = f(points)
    our_initial_points.append(point)
    mean_f_on_initial_points.append(np.mean(points_f))
    var_f_on_initial_points.append(np.var(points_f))



our_initial_points = np.array(our_initial_points)
mean_f_on_initial_points = np.array(mean_f_on_initial_points)
var_f_on_initial_points = np.array(var_f_on_initial_points)

K = np.exp(-gamma*np.abs(our_initial_points[..., np.newaxis] - our_initial_points[np.newaxis, ...]))


x_s = []
mus = []
vars = []

for i in range(100):
    x = i
    y = np.exp(-gamma*np.abs(our_initial_points - x))
    y_c = np.concatenate((y,[1]), axis=0)

    # K_n = np.concatenate((K,y[..., np.newaxis]), axis=1)
    #
    # K_n = np.concatenate((K_n,y_c[np.newaxis, ...]), axis=0)

    m = y.dot(np.linalg.inv(K)).dot(mean_f_on_initial_points)
    v = -y.dot(np.linalg.inv(K)).dot(y) + 1

    x_s.append(x)
    mus.append(m)
    vars.append(v)



x_s = np.array(x_s)

fig, axs = plt.subplots()

axs.errorbar(x_s, mus,yerr=np.sqrt(vars))
plt.plot(x_s,f_w_n(x_s),c='red')
# plt.plot(x,f_w_n(x), c='r')


for v in our_initial_points:
    plt.axvline(x=v,color='g', linewidth=0.1)


plt.show()