import tensorflow as tf
import numpy as np


def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.
    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.
    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)






data = np.random.normal(10,3,size=(30,250,32))


data_x = np.expand_dims(data, axis=2)
data_y = np.expand_dims(data, axis=1)
result = np.sum((data_x - data_y)**2, axis=-1)

print(data_x.shape)
print(data_y.shape)


result_2 = euclidean_squared(tf.convert_to_tensor(data), tf.convert_to_tensor(data))


diff = np.sum(result - result_2)

print(diff)

print(result.shape)