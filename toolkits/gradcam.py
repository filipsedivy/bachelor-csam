import numpy as np
import tensorflow as tf
from keras import Model


def grad_cam(model, img, idx: int = None, layer_name: str = 'mixed10'):
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model(
        [model.inputs],
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(img_tensor)

        if idx is None:
            idx = np.argmax(predictions[0])

        output = predictions[:, idx]
        gradients = tape.gradient(output, conv_output)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(
        tf.multiply(pooled_gradients, conv_output),
        axis=-1
    )

    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)

    if max_heat == 0:
        max_heat = 1e-10

    heatmap /= max_heat

    return np.squeeze(heatmap)


def grad_cam_plus(model, img, idx: int = None, layer_name: str = 'mixed10'):
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model(
        [model.inputs],
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_output, predictions = heatmap_model(img_tensor)

                if idx is None:
                    idx = np.argmax(predictions[0])

                output = predictions[:, idx]
                conv_first_grad = tape3.gradient(output, conv_output)
            conv_second_grad = tape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = tape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha = np.where(alpha != 0.0, alpha, 1e-10)

    alphas = alpha_num / alpha

    weights = np.maximum(conv_first_grad[0], 0.0)

    weights_sum = np.sum(weights * alphas, axis=(0, 1))

    grad_cam_map = np.sum(weights_sum * conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap
