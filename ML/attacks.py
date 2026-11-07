import numpy as np
import tensorflow as tf

def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, steps=40):
    images = tf.convert_to_tensor(images)

    # Ensure labels are integer class indices
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    labels = tf.convert_to_tensor(labels)

    adv_images = tf.identity(images)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

        gradient = tape.gradient(loss, adv_images)
        signed_grad = tf.sign(gradient)
        adv_images = adv_images + alpha * signed_grad

        # Project the perturbations back into the epsilon-ball of the original image
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 1)  # keep pixel values valid

    return adv_images.numpy()

def fgsm_attack(model, images, labels, epsilon=0.3):
    images = tf.convert_to_tensor(images)

    # Ensure labels are integer class indices
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    labels = tf.convert_to_tensor(labels)

    adv_images = tf.identity(images)

    with tf.GradientTape() as tape:
        tape.watch(adv_images)
        predictions = model(adv_images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    gradient = tape.gradient(loss, adv_images)
    signed_grad = tf.sign(gradient)

    adversarial_image = adv_images + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    return adversarial_image.numpy()
