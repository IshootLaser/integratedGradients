import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_compare(imgs1, imgs2, imgs3, labels):
    """
    Plot two lists of images side by side to compare visualization
    Args:
        imgs1: Original images as a list of np array
        imgs2: XAI images as a list of np array
        imgs3: XAI images highlights as a list of np.array
        labels: top 5 predicted labels for each image
    """
    assert len(imgs1) == len(imgs2), 'The length of 2 inputs must be the same!'
    length = len(imgs1)
    f, ax = plt.subplots(length, 4, figsize = (10, len(imgs1) * 2))
    for i in range(length):
        ax[i, 1].imshow(imgs1[i])
        ax[i, 1].axis('off')
        ax[i, 2].imshow(imgs2[i])
        ax[i, 2].axis('off')
        ax[i, 3].imshow(imgs3[i])
        ax[i, 3].axis('off')
        label = [x[1] for x in labels[i]]
        w = [x[-1] for x in labels[i]]
        ax[i, 0].barh(np.arange(len(labels[i]), 0, -1), w, tick_label = label)
        if i == 0:
            ax[i, 1].set_title('original')
            ax[i, 2].set_title('XAI images')
            ax[i, 3].set_title('top pixels')
    plt.show()
    return

def overlay(img, mask, ix = None):
    """
    Visualize img and mask together in 1 picture.
    Args:
        img: the original image as np array of shape (h, w, 3)
        mask: the mask of shape(h, w)
    Returns:
        An image with mask overlay
    """
    if mask.dtype == np.float32:
        assert (mask.max() <= 1) & (mask.min() >= 0), 'the mask should be a float between 0 and 1'
        mask = np.expand_dims(mask, axis = -1)
        img = img.astype(np.float32)
        img = (img * mask).astype(np.uint8)
    elif mask.dtype == np.uint8:
        x, y = np.where(mask != 0)
        img = img.copy()
        if ix is None:
            img[x, y, :] = np.array([0, 255, 0])
        else:
            color = np.array([0, 0, 0])
            color[ix] = 255
            img[x, y, :] = color
        img = img.astype(np.uint8)
    return img

def normalize(gradient,
              high_percentile = 99.9,
              low_percentile = 70,
              high_bound = 1,
              low_bound = 0.2):
    """
    Normalize a floating point array range ~ [0, 1]
    Args:
        gradient: the aggregated gradient of shape (h, w)
        high_percentile: upper bound of percentile cap
        low_percentile: lower bound of percentile cap
        high_bound: where to map the upper percentile cap
        low_bound: where to map the lower percentile cap
    Returns:
        scale: a float array range ~ [low_bound, high_bound] of shape(h, w)
    """
    assert (high_bound <= 1) & (low_bound >= 0), 'The scale range is [0, 1]'
    gradient_flat = gradient.reshape(-1)
    high_thres = np.percentile(gradient, high_percentile)
    low_thres = np.percentile(gradient, low_percentile)
    gradient = np.clip(gradient, low_thres, high_thres)
    # scale to between 0 and 1
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    # shift it to range (low_bound, high_bound)
    scale = gradient * (high_bound - low_bound) + low_bound
    return scale

def selectTopPercent(gradient, percent = 0.1):
    """
    Select the top x percent pixels in the integrated gradient image
    Args:
        gradient: the integrated gradient image of shape(h, w)
        percent: float between 0 and 1
    Returns:
        mask: a mask that indicates the top x percent pixel attribution
    """
    grad_flat = gradient.reshape(-1)
    percent = (1 - percent) * 100
    thres = np.percentile(grad_flat, percent)
    x, y = np.where(gradient >= thres)
    mask = np.zeros(gradient.shape).astype(np.uint8)
    mask[x, y] = 255
    return mask

def plot_sbs(img1, img2):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[1].imshow(img2)
    ax[1].axis('off')
    f.suptitle('input vs. baseline')
    plt.show()

def visualize(ig, img):
    binary_mask = selectTopPercent(ig)
    ovl = overlay(img, ig)
    highlight = overlay(img, binary_mask)
    return binary_mask, ovl, highlight

def visualize_clusters(ig):
    blk = np.zeros((224, 224, 3)).astype(np.uint8)
    for n, j in enumerate(ig):
        j = selectTopPercent(j, 0.05)
        y, x = np.where(j != 0)
        color = np.array([0, 0, 0])
        color[n] = 255
        blk[y, x] = color
    return blk[..., ::-1]