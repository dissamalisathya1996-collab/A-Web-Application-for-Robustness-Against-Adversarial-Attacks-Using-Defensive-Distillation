import numpy as np
from PIL import Image
import tensorflow as tf
from ML.attacks import fgsm_attack, pgd_attack

def preprocess_image(img_array):
    """For displaying images (unnormalized and resized for viewing)"""
    img = np.squeeze(img_array) * 255.0
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype('uint8'))
    img = img.resize((224, 224), Image.NEAREST)  # Resize to 224x224 for clearer display
    return img


def get_prediction_confidence(model, image):
    """Return top-1 predicted label and confidence"""
    image = np.expand_dims(image, axis=0)  # [1, H, W, C]
    image = image.astype('float32') / 255.0  # Normalize to [0, 1] if not already
    preds = model.predict(image, verbose=0)
    label = np.argmax(preds[0])
    confidence = np.max(preds[0])
    return label, confidence


def apply_attack(image, model, attack_type, label=None):
    """Applies the selected attack to the image."""
    if label is None:
        label, _ = get_prediction_confidence(model, image)

    label = np.array([label])  # ensure it's a batch label
    image = np.expand_dims(image, axis=0)  # [1, H, W, C]

    # Normalize the image
    image = image.astype('float32') / 255.0

    if attack_type == 'fgsm':
        attacked = fgsm_attack(model, image, label, epsilon=0.3)  # You can increase epsilon to 0.3
    elif attack_type == 'pgd':
        attacked = pgd_attack(model, image, label, epsilon=0.3, alpha=0.01, steps=40)
    else:
        return np.squeeze(image)  # no attack

    return np.squeeze(attacked)

