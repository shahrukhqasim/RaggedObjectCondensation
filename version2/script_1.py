import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from version2.the_model import OverfittingKing
from version2.ops import *





"""
This script is a toy script for implementation of object condensation method.

We will generate a batch of points in a 2D input space. Then we will transform these points into another clustering
space using a vertex invariant dense neural network. The idea is to overfit the points so to show that the loss is
working.

On y va!

"""


random.seed(100)
np.random.seed(100)



size_batch = 3

# Num objects will [min_num_objects, max_num_objects)
min_num_objects = 2
max_num_objects = 4


# Num vertices per object will be [minimum_vertices_per_object, maximum_vertices_per_object)
minimum_vertices_per_object = 5
maximum_vertices_per_object = 7

# Num background vertices will be [minimum_background_vertices, maximum_background_vertices)
minimum_background_vertices = 3
maximum_background_vertices = 9


mininum_x_value = 0.
maximum_x_value = 1000.
# Not being used yet: (just easier this way)
minimum_y_value = 0.
maximum_y_value = 1000.



input_data = []
classes = []
row_splits = [0]

for i in range(size_batch):
    num_objects_this_batch = np.random.randint(min_num_objects, max_num_objects)

    vertices_input_space = []
    vertices_classes = []

    # Generate all the objects
    for i in range(num_objects_this_batch):
        num_vertices_this_object = np.random.randint(minimum_vertices_per_object, maximum_vertices_per_object)
        vertices_this_object = np.random.uniform(mininum_x_value, maximum_x_value, size=(num_vertices_this_object, 2)).astype(np.float32).tolist()
        object_class = (np.ones(shape=num_vertices_this_object) * i).astype(np.float32).tolist()

        vertices_input_space += vertices_this_object
        vertices_classes += object_class

    # Generate the background
    num_vertices_background = np.random.randint(minimum_background_vertices, maximum_background_vertices)
    vertices_background = np.random.uniform(mininum_x_value, maximum_x_value,
                                             size=(num_vertices_background, 2)).astype(np.float32).tolist()
    object_class = (np.ones(shape=num_vertices_background) * (-1)).astype(np.float32).tolist()

    vertices_input_space = vertices_background + vertices_input_space
    vertices_classes = object_class + vertices_classes

    # Convert everything to numpy array for plotting
    n_vertices_input_space = np.array(vertices_input_space)
    n_vertices_classes = np.array(vertices_classes)

    # Plot it maybe? Should be rather simple!!!
    s = (n_vertices_classes==-1)*0.1 + (n_vertices_classes!=-1)*3
    # plt.scatter(x=n_vertices_input_space[:,0], y=n_vertices_input_space[:,1], c = (n_vertices_classes+1), s=s, cmap=plt.get_cmap('Dark2'))
    # plt.show()

    row_splits += [row_splits[-1]+len(vertices_classes)]
    input_data += vertices_input_space
    classes += vertices_classes

# Let's construct the model which is basically nothing
model = OverfittingKing()

print(np.random.uniform(0,100,size=(1000,2)).shape)
# Initialize the weights etc
model.call(np.random.uniform(0,100,size=(1000,2)))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def plot_stuff(vertices, vertices_classes, title):
    s = (vertices_classes==-1)*0.1 + (vertices_classes!=-1)*6
    plt.title(title)
    plt.scatter(x=vertices[:,0], y=vertices[:,1], c = (vertices_classes+1), s=s, cmap=plt.get_cmap('Dark2'))
    plt.show()





tf_input = tf.convert_to_tensor(input_data)
tf_classes = tf.convert_to_tensor(classes)
tf_row_splits = tf.convert_to_tensor(row_splits)
tf_row_splits = tf.cast(tf_row_splits, tf.int32)

writer = tf.summary.create_file_writer("summaries_def/the_one")
with writer.as_default():
    iteration = 0
    for iteration in range(100000):
        with tf.GradientTape() as tape:
            output_space, beta_values = model(tf_input)
            beta_values = beta_values[:, 0]

            # plot_stuff(output_space.numpy()[0:row_splits[1]], tf_classes.numpy()[0:row_splits[1]], "Iteration "+str(iteration))

            loss_value = evaluate_loss(output_space, beta_values, tf_classes, tf_row_splits)
            print("Wowww")
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        tf.summary.scalar("loss", loss_value, step=iteration)
        loss_value = float(loss_value)
        print(iteration, loss_value)
        iteration += 1




