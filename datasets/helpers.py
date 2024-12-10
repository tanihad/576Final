import numpy as np


BACKGROUND_LABEL = 28

def tens2image(im):
    tmp = np.squeeze(im.cpu().numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def convert_to_coco_format(x1, y1, x2, y2):
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    return x, y, width, height


def convert_from_coco_format(x, y, width, height):
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return x1, y1, x2, y2