import numpy as np




x1 = np.random.normal(10,3,size=(100,2))
y1 = np.zeros(shape=(100,))+1
x2 = np.random.normal(8,2,size=(100,2))
y2 = np.zeros(shape=(100,))+2
x3 = np.random.normal(3,1,size=(100,2))
y3 = np.zeros(shape=(100,))+3


X = np.concatenate((x1,x2,x3), axis=0)
Y = np.concatenate((y1,y2,y3), axis=0)

gamma = 0.8


data_x = np.expand_dims(X, axis=1)
data_y = np.expand_dims(X, axis=0)
blabla = np.sum((data_x - data_y)**2, axis=-1)
result = np.exp(-blabla*gamma)

w = np.linalg.inv(result).dot(Y)


y = np.sum(result*w[np.newaxis, ...], axis=1)


print(y.shape)


print(y)