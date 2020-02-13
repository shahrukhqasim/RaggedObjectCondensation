import numpy as np
import tensorflow as tf



our_shower_indices = np.array([ 2,  3,  4,  2, -1, -1,  4, -1, -1,  4,  0,  3,  4,  0,  3,  0,  1,
        1,  1,  4,  4,  2])



our_features = np.array([[7.23, 4.9 , 8.54],
       [4.22, 6.63, 1.04],
       [6.31, 6.68, 6.39],
       [8.06, 9.24, 4.51],
       [6.6 , 9.57, 3.46],
       [6.01, 7.44, 8.79],
       [1.9 , 3.16, 8.98],
       [0.42, 4.53, 3.44],
       [5.4 , 4.77, 2.82],
       [5.38, 1.67, 9.63],
       [5.57, 9.84, 9.27],
       [2.78, 4.31, 8.35],
       [2.17, 1.48, 7.32],
       [1.13, 7.93, 8.05],
       [3.03, 5.54, 2.66],
       [1.41, 2.45, 0.27],
       [6.76, 4.33, 2.2 ],
       [0.04, 8.07, 4.36],
       [6.16, 7.75, 1.07],
       [2.56, 0.57, 3.45],
       [7.46, 6.75, 8.06],
       [4.86, 1.45, 4.63]])



our_features_distance = np.zeros_like(our_features)

# print(our_features[np.argwhere(our_features==0)])

# print("Hello world")
#
# exit()

sum = 0
for i in range(5+1):
    # print("Our features shape", our_features.shape)
    # print("This shower indices shape", ((our_shower_indices+1)==i).shape)
    this_shower_mean = np.mean(our_features[np.argwhere(our_shower_indices+1==i)], axis=0)[np.newaxis, :]

    # print(i, np.where((our_shower_indices+1)==i))


    our_features_distance[(our_shower_indices+1)==i] = our_features[(our_shower_indices+1)==i] - this_shower_mean





print(our_features_distance)

# our_features_dist = np.array([[ 2.95      ,  2.555     ,  1.47      ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [-0.3075    ,  4.8375    , -1.0525    ],
#        [ 0.08      , -0.43714286,  5.15428571],
#        [ 0.        ,  0.        ,  0.        ],
#        [-1.5475    ,  1.3475    ,  1.5975    ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.78      ,  0.78285714, -0.23571429],
#        [-0.37      , -2.42714286, -4.24571429],
#        [-0.9775    , -1.9625    , -0.7025    ],
#        [-2.61      , -0.78714286, -3.83571429],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 1.13      ,  5.53285714, -1.87571429],
#        [ 2.8325    , -4.2225    ,  0.1575    ],
#        [ 4.61      , -0.26714286,  1.85428571],
#        [ 0.        ,  0.        ,  0.        ],
#        [-3.62      , -2.39714286,  3.18428571],
#        [-2.95      , -2.555     , -1.47      ]])


