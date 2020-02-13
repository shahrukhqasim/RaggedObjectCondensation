import tensorflow as tf
import numpy as np




############################################################################################################
###### ###### ###### ###### ######   Only to produce sample data   ###### ###### ###### ###### ###### ######
############################################################################################################
sum_vertices_max = 300
num_max_showers = 4
num_vertices_shower_min = 20
num_vertices_shower_max = 30


min_showers = 2
max_showers=6


split = 0

row_splits = [0]
shower_indices = []
while True:
    this_shower_num_vertices = num_vertices_shower_min + np.random.randint(num_vertices_shower_max - num_vertices_shower_min)

    if split + this_shower_num_vertices > sum_vertices_max:
        break


    split += this_shower_num_vertices
    row_splits.append(split)

    our_max_showers = min_showers + np.random.randint(max_showers-min_showers)

    shower_indices += np.random.randint(our_max_showers+1, size=this_shower_num_vertices).tolist()


beta_values = np.random.randint(0, 1000, size=len(shower_indices)).astype(np.float)/10000


# The following two variables are interesting
ragged_tensor_beta_values = tf.RaggedTensor.from_row_splits(beta_values, row_splits=row_splits)
ragged_tensor_shower_indices = tf.cast(tf.RaggedTensor.from_row_splits(shower_indices, row_splits=row_splits), tf.int64)

############################################################################################################


print(ragged_tensor_shower_indices)

# Multiply row ids by a huge number and then add it so we do argmax within each batch element
sorting_tensor = ragged_tensor_shower_indices.values + ragged_tensor_shower_indices.value_rowids()*40000000
sorting_tensor = tf.argsort(sorting_tensor)[..., tf.newaxis]

# Now the second dimension is sorted by shower index
ragged_tensor_beta_values = tf.RaggedTensor.from_row_splits(tf.gather_nd(ragged_tensor_beta_values.values, indices=sorting_tensor), row_splits=row_splits)
ragged_tensor_shower_indices = tf.RaggedTensor.from_row_splits(tf.gather_nd(ragged_tensor_shower_indices.values, indices=sorting_tensor), row_splits=row_splits)


print(ragged_tensor_shower_indices)


# make row splits according to number of showers in each of the batch element
row_splits_secondary =tf.cumsum(tf.concat(([0],1+tf.reduce_max(ragged_tensor_shower_indices, axis=1)), axis=0))
print(row_splits_secondary)

additive = (row_splits_secondary[0:-1])[..., tf.newaxis]
ragged_tensor_shower_indices_across_all_batch_elements = ragged_tensor_shower_indices + additive
print(ragged_tensor_shower_indices_across_all_batch_elements)


ragged_tensor_shower_indices_across_all_batch_elements = ragged_tensor_shower_indices_across_all_batch_elements.values

showers_ragged_indices_only = tf.RaggedTensor.from_value_rowids(values=ragged_tensor_shower_indices.values, value_rowids=ragged_tensor_shower_indices_across_all_batch_elements)
ragged_tensor_shower_indices = tf.RaggedTensor.from_row_splits(values=showers_ragged_indices_only, row_splits=row_splits_secondary)

print(ragged_tensor_shower_indices)



showers_ragged_beta_values = tf.RaggedTensor.from_value_rowids(values=ragged_tensor_beta_values.values, value_rowids=ragged_tensor_shower_indices_across_all_batch_elements)
ragged_tensor_beta_values = tf.RaggedTensor.from_row_splits(values=showers_ragged_beta_values, row_splits=row_splits_secondary)

print(ragged_tensor_beta_values.shape)

0/0


# TODO: Apparently argmin doesn't work with ragged tensors arghh so annoying
# tf.argmin(ragged_tensor_shower_indices, axis=2)
# Find the minimum by
print(tf.reduce_max(ragged_tensor_beta_values, axis=-1))

