from scipy.signal import convolve2d
import scipy.ndimage.filters
import numpy as np
from imageio import imread
from skimage.color import rgb2gray

GRAY_MAX_VALUE = 255

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def build_binomial_array(binomial_size):
    """
    This function returns a 1D array which is the binomial array
    including binomial size coefficients.
    :param binumial_size: The number of binomial coefficients
    :return:  1D array which is the binomial array
    including binomial size coefficients.
    """
    if binomial_size == 1:
        return np.array([1])

    base_conv = np.array([1, 1]).reshape(1,2)
    filter_array = np.array([1, 1]).reshape(1,2)

    while filter_array.size < binomial_size:
        filter_array = convolve2d(filter_array, base_conv)

    return filter_array.astype(np.float64)


def reduce_image(filter, im):
    """
    This function reduces the size of a given image after blurring it with filter.
    :param filter: The filter to blur the image with
    :param new_im: a RGB or grayscale image
    :return: The reduced image
    """
    x = im.shape[0] // 2
    y = im.shape[1] // 2

    if len(im.shape) == 3:
        new_im = np.zeros((x, y, 3), dtype=im.dtype)

        for i in range(3):
            blurred_color = blur(im[:, :, i], filter)
            new_im[:, :, i] = (blurred_color[::2, ::2])
        return new_im
    elif len(im.shape) == 2:
        blurred_im = blur(im, filter)
        new_im = blurred_im[::2, ::2]

        return new_im

def blur(im, filter_1D):
    """
    This function blurs the image with a gaussian filter
    :return: the image after it was blur
    """
    im = scipy.ndimage.filters.convolve(im, filter_1D)

    return scipy.ndimage.filters.convolve(im, filter_1D.T)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function constructs a gaussian_pyramid from the given image with
    max_levels levels and gaussian filter size at filter_size.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return: pyr, filter_vec.
    pyramid pyr as a standard python array (i.e. not numpy’s array)
    with maximum length of max_levels, where each element of the array is a
    grayscale image.
    The functions should also output filter_vec which is row vector of shape
    (1, filter_size) used for the pyramid construction.
    This filter should be built using a consequent 1D convolutions of [1 1]
    with itself in order to derive a row of the binomial coefficients which
    is a good approximation to the Gaussian profile.
    The filter_vec is be normalized.
    """
    new_im = np.copy(im)

    gaussian_pyramid = [im]

    filter_1D = build_binomial_array(filter_size)

    normalize_coefficient = np.sum(filter_1D)

    filter = (filter_1D /normalize_coefficient)

    # main loop
    while new_im.shape[0] >= 16 and new_im.shape[1] >= 16 and \
            len(gaussian_pyramid) < max_levels:

        new_im = reduce_image(filter, new_im)

        gaussian_pyramid.append(new_im)

    return gaussian_pyramid, filter

def read_image(filename, representation):
    """
    3.1 Reading an image into a given representation.
    :param filename: read_image(filename, representation).
    :param representation: representation code, either 1 or 2 defining
    whether the output should be a grayscale image (1) or an RGB image (2).
    If the input image is grayscale, we won’t call it with representation = 2.
    :return: This function returns an image, make sure the output image
    is represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities)
    normalized to the range [0, 1].
    """
    im = (imread(filename).astype(np.float64)) / GRAY_MAX_VALUE

    if representation == 1:

        return rgb2gray(im)

    elif representation == 2:

        return (im)