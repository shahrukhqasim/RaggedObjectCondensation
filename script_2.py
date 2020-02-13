import tensorflow as tf




a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
i = [0,4,4,11]


add = tf.constant([1,2,3])
add = add[..., tf.newaxis]

rt = tf.RaggedTensor.from_row_splits(a, row_splits=i)
print(rt)
rt = rt+add
print(rt)




a = [1,1,1,1,2,2,2,2,2,2]
d = [4,4,1,1,2,4,5,6,7,8.]


rt = tf.RaggedTensor.from_value_rowids(values=d, value_rowids=a)

# rt = tf.concat([rt,rt], axis=0)

# rt = rt[:, 1:]

#
# print(tf.reduce_max(rt[:, 1:], axis=-1))
#
# # print(rt[:, 1:])
#
# print(rt.shape)

print(rt[1:].value_rowids())

print(rt.shape)