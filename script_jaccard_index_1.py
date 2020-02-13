import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def lovasz_grad(gt_sorted, preds, errors):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)

    gts = tf.reduce_sum(gt_sorted)

    intersection = gts - tf.math.cumsum(gt_sorted)
    union = gts + tf.math.cumsum(1 - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.where(tf.math.is_inf(jaccard), 0, jaccard)

    if p > 1:  # cover 1-pixel case
        jaccard_2 = jaccard[1:p] - jaccard[0:-1]


    jaccard = tf.concat([[jaccard[0]], jaccard_2], axis=0)

    errors_1 = errors[..., tf.newaxis]
    errors_2 = errors[tf.newaxis, ...]

    tiled_gt = tf.tile(gt_sorted[tf.newaxis, ...], multiples=[p, 1])
    tiled_pred = tf.tile(preds[tf.newaxis, ...], multiples=[p, 1])



    # print(errors.numpy().tolist())
    #
    #
    # print((errors_2<errors_1[1, :]).numpy().tolist())
    #
    # print(tiled_gt[0,:].numpy().tolist())
    # print(tiled_pred[0,:].numpy().tolist())


    tiled_gt = tf.where(errors_2<errors_1, 0, tiled_gt)
    tiled_pred = tf.where(errors_2<errors_1, 0, tiled_pred)

    #
    # print(tiled_gt[:,0].numpy().tolist())
    # print(tiled_pred[:,0].numpy().tolist())
    #
    # print(tf.reduce_sum(tiled_gt), tf.reduce_sum(tiled_pred))
    #
    # 0/0


    #
    # print(tf.math.count_nonzero(tiled_gt, axis=1).numpy().tolist())
    #
    #
    # 0/0


    # print(tiled_gt.dtype)

    secondary = 1 - tf.reduce_sum(tiled_gt*tiled_pred, axis=1) / tf.reduce_sum(tf.math.minimum(tiled_gt + tiled_pred, 1), axis=1)
    secondary_2 = secondary[1:p] - secondary[0:-1]
    secondary = tf.concat([[secondary[0]], secondary_2], axis=0)
    secondary = tf.where(tf.math.is_nan(secondary), 0, secondary)
    secondary = tf.where(tf.math.is_inf(secondary), 0, secondary)


    print(secondary)

    print(jaccard)


    #
    # print(errors)
    #
    # # print(tf.reduce_sum(secondary, axis=1))
    #
    #
    #
    #
    # print(secondary.numpy().tolist())
    #
    # print(jaccard.numpy().tolist())
    # print(tf.reduce_sum(secondary), tf.reduce_sum(jaccard))
    #

    # 0/0


    return secondary


def lovasz_hinge_flat(logits, labels, preds):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)
    preds = tf.cast(preds, tf.float32)

    signs = 2. * labels - 1.
    errors = (1. - logits * signs)

    perm = tf.argsort(errors, axis=0, direction='DESCENDING')[..., tf.newaxis]

    errors_sorted = tf.gather_nd(errors, perm)
    gt_sorted = tf.gather_nd(labels, perm)


    grad = lovasz_grad(gt_sorted, tf.gather_nd(preds, perm), errors_sorted)


    loss = tf.tensordot(tf.nn.relu(errors_sorted), grad, 1)



    return loss



def generate_circle(height, width, center_x, center_y, radius):
    ground_truth = np.zeros(shape=(height, width))

    locations = np.zeros(shape=(height, width, 2))
    locations[:,:,0] = np.tile(np.arange(0, height)[..., np.newaxis], reps=[1, width])
    locations[:,:,1] = np.tile(np.arange(0, width)[np.newaxis, ...], reps=[height, 1])

    ground_truth[(locations[:,:,0]-center_y)**2+(locations[:,:,1]-center_x)**2<radius*radius] = 1
    return ground_truth

def generate_circle(height, width, center_x, center_y, radius):
    ground_truth = np.zeros(shape=(height, width))

    locations = np.zeros(shape=(height, width, 2))
    locations[:,:,0] = np.tile(np.arange(0, height)[..., np.newaxis], reps=[1, width])
    locations[:,:,1] = np.tile(np.arange(0, width)[np.newaxis, ...], reps=[height, 1])

    ground_truth[(locations[:,:,0]-center_y)**2+(locations[:,:,1]-center_x)**2<radius*radius] = 1
    return ground_truth


ground_truth = generate_circle(100, 100, 50,50, 10)
# ground_truth[ground_truth==0] = -1


loss_values = []
iou_values = []
normal_loss_values = []


for i in range(11):
    predictionx = generate_circle(100, 100, 30+i*2,50, 10)
    prediction = predictionx.copy()
    prediction[predictionx==0] = -1
    prediction = prediction*3

    # plt.imshow(prediction)
    # plt.show()

    pp = prediction.reshape(-1) + np.random.normal(0, 0.003, size=prediction.size)
    gg = ground_truth.reshape(-1)
    loss = lovasz_hinge_flat(pp, gg, predictionx.reshape(-1))
    loss = float(loss)

    iou = np.sum(predictionx * ground_truth) / np.sum((predictionx + ground_truth)>0)
    iou_values.append(iou)

    loss_values.append(loss)
    normal_loss_values.append(np.mean(predictionx!=ground_truth))


print(loss_values)
print(iou_values)
print(normal_loss_values)

corr = np.corrcoef(loss_values, iou_values)
corr2 = np.corrcoef(normal_loss_values, iou_values)

print(corr, corr2)