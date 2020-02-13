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

    new_shower_indices = np.random.randint(our_max_showers+1, size=this_shower_num_vertices)


    zero_indices = np.random.choice([0,1], p=[0.95, 0.05], size=this_shower_num_vertices)==1

    new_shower_indices[zero_indices] = -1
    new_shower_indices = new_shower_indices.tolist()
    shower_indices += new_shower_indices


print(shower_indices)


n_features = 3

beta_values = np.random.randint(0, 1000, size=len(shower_indices)).astype(np.float)/10000
feature_space = np.random.randint(0, 1000, size=(len(shower_indices), n_features)).astype(np.float)/100


shower_indices = np.array(shower_indices)

############################################################################################################
loss_batch_elements = []
loss_2_batch_elements = []


loss_data = []

print(tf.constant(0))

for i in range(len(row_splits)-1):
    our_shower_indices = shower_indices[row_splits[i]:row_splits[i+1]]
    our_max_showers = np.max(our_shower_indices)+1
    our_features = feature_space[row_splits[i]:row_splits[i+1]]
    our_features_distance = np.zeros_like(our_features)
    our_features_distance_repul = np.zeros_like(our_features)
    shower_means = []
    sum = 0
    for j in range(our_max_showers+1):
        this_shower_mean = np.mean(our_features[np.argwhere(our_shower_indices+1==j)], axis=(0,1))
        shower_means.append(this_shower_mean)

        this_shower_mean = this_shower_mean[np.newaxis, :]

        our_features_distance[(our_shower_indices+1)==j] = np.abs(our_features[(our_shower_indices+1)==j] - this_shower_mean)

    this_batch_element_sum = np.sum(our_features_distance, axis=1)
    loss_data.append(this_batch_element_sum)

    repul_sum = 1 - this_batch_element_sum
    repul_sum[repul_sum<0] = 0


    this_batch_element_sum = np.mean(this_batch_element_sum)
    loss_batch_elements.append(this_batch_element_sum)
    loss_2_batch_elements.append(np.mean(repul_sum))


print("Ground truth values", np.mean(loss_batch_elements), np.mean(loss_2_batch_elements))


# The following two variables are interesting
ragged_tensor_feature_space = tf.RaggedTensor.from_row_splits(feature_space, row_splits=row_splits)
ragged_tensor_shower_indices = tf.cast(tf.RaggedTensor.from_row_splits(shower_indices, row_splits=row_splits), tf.int32)

# Depth will be variable now which is pretty cool!

one_hot_values = tf.RaggedTensor.from_row_splits(tf.one_hot(ragged_tensor_shower_indices.values +1, depth=(tf.reduce_max(ragged_tensor_shower_indices)+2)), row_splits)
one_hot_values = tf.cast(one_hot_values, tf.float64)

expanded_feature_space = tf.expand_dims(ragged_tensor_feature_space, axis=2) * tf.expand_dims(one_hot_values, axis=3)
mean_features = tf.reduce_sum(expanded_feature_space, axis=1) / tf.expand_dims(tf.reduce_sum(one_hot_values, axis=1), axis=2)

mean_features = tf.where(tf.math.is_nan(mean_features), 0, mean_features)



# :( This doesn't work its not supported in ragged tensors:
# distance_from_mean = mean_features - expanded_feature_space

# But fortunately, the we can do gather_nd in a very straightforward way using gather nd
distance_from_mean = expanded_feature_space - tf.RaggedTensor.from_row_splits(tf.gather_nd(mean_features, expanded_feature_space.value_rowids()[..., tf.newaxis]), row_splits)
# Mask it so we are not taking distances from zeros!
distance_from_mean = distance_from_mean * one_hot_values[..., tf.newaxis]


# Sum loss over all the features
distance_from_mean = tf.reduce_sum(tf.math.abs(distance_from_mean), axis=(-1,-2))


# Mean over vertices
distance_from_mean = tf.reduce_mean(distance_from_mean, axis=1)
# Mean over the batch elements
distance_from_mean = tf.reduce_mean(distance_from_mean)



# Same thing for repulusion
repulsion_from_mean = expanded_feature_space - tf.RaggedTensor.from_row_splits(tf.gather_nd(mean_features, expanded_feature_space.value_rowids()[..., tf.newaxis]), row_splits)
# Mask it so we are not taking distances from zeros!
repulsion_from_mean = repulsion_from_mean * one_hot_values[..., tf.newaxis]


# Sum loss over all the features
repulsion_from_mean = 1 - tf.reduce_sum(tf.math.abs(repulsion_from_mean), axis=(-1,-2))
# Dont kill me for this:
repulsion_from_mean = tf.RaggedTensor.from_row_splits(tf.nn.relu(repulsion_from_mean.values), row_splits)

# Mean over vertices
repulsion_from_mean = tf.reduce_mean(repulsion_from_mean, axis=1)
# Mean over the batch elements
repulsion_from_mean = tf.reduce_mean(repulsion_from_mean)



print("TF values", distance_from_mean, repulsion_from_mean)