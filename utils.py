from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import cv2

def plot_compare(imgs1, imgs2):
    """
    Plot two lists of images side by side to compare visualization
    Args:
        imgs1: Original images as np array
        imgs2: XAI images as np array
    """
    assert len(imgs1) == len(imgs2), 'The length of 2 inputs must be the same!'
    length = len(imgs1)
    f, ax = plt.subplots(length, 2)
    for i in range(length):
        ax[i, 0].imshow(imgs1[i])
        ax[i, 0].axis('off')
        ax[i, 1].imshow(imgs2[i])
        ax[i, 1].axis('off')
        if i == 0:
            ax[i, 0].set_title('original')
            ax[i, 1].set_title('XAI images')
    f.suptitle('Comparison')
    plt.show()
    return

def overlay(img, mask, mode = 'mask'):
    """
    Visualize img and mask together in 1 picture.
    Args:
        img: the original image as np array
        mask: the binary mask produced by integrated gradients
        mode: either 'mask' or 'highlight'
    Returns:
        An image with mask overlay
    """
    if mode not in ['mask', 'highlight']:
        raise ValueError("mode must be either 'mask' or 'highlight'")
    if mode == 'mask':
        img = cv2.bitwise_and(img, img, mask = mask)
    else:
        x, y = np.where(mask != 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img[x, y, :] = np.array([255, 0, 0])
    return img

def normalize(img):
    """
    Normalize a floating point array to unsign 8-bit image
    """
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)

def selectTopPercent(img, percent = 0.05):
    """
    Select the top x percent pixels in the integrated gradient image
    Args:
        img: the integrated gradient image
        percent: float between 0 and 1
    Returns:
        
    """
