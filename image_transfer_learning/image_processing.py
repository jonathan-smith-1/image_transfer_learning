
def make_square(img):
    """
    Trim an image to make it square by keeping the centre of the original image.

    Args:
        img (Numpy array): Input image with shape (height, width, channels)

    Returns:
        Numpy array of trimmed image.

    """
    height, width, channels = img.shape

    if height >= width:
        h_max = int(height/2 + width/2)
        h_min = int(height/2 - width/2)

        return img[h_min:h_max, :, :].copy()

    else:
        w_max = int(width/2 + height/2)
        w_min = int(width/2 - height/2)

        return img[:, w_min:w_max, :].copy()
