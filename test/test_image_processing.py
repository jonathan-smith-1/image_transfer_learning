import numpy as np
from image_transfer_learning.image_processing import make_square


def test_odd_height_odd_width():

    """
    WHEN: There is an image with odd height, an odd width (different values) and 1 or 3 channels
    AND: The make_square function is called
    WHAT SHOULD HAPPEN: A square image is returned.
                        Number of channels is preserverd.

    Returns:
        Nothing

    """

    in_height = 5
    in_width = 3

    for in_channels in [1, 3]:

        shape = in_height, in_width, in_channels

        input_image = np.zeros(shape)

        output_image = make_square(input_image)

        out_height, out_width, out_channels = output_image.shape

        assert out_height == out_width
        assert out_channels == in_channels


def test_odd_height_even_width():
    """
    WHEN: There is an image with odd height, an even width and three channels
    AND: The make_square function is called
    WHAT SHOULD HAPPEN: A square image is returned.
                        Number of channels is preserverd.

    Returns:
        Nothing

    """

    in_height = 5
    in_width = 4

    for in_channels in [1, 3]:

        shape = in_height, in_width, in_channels

        input_image = np.zeros(shape)

        output_image = make_square(input_image)

        out_height, out_width, out_channels = output_image.shape

        assert out_height == out_width
        assert out_channels == in_channels


def test_even_height_even_width():
    """
    WHEN: There is an image with even height, an even width (different values) and three channels
    AND: The make_square function is called
    WHAT SHOULD HAPPEN: A square image is returned.
                        Number of channels is preserverd.

    Returns:
        Nothing

    """

    in_height = 4
    in_width = 6

    for in_channels in [1, 3]:

        shape = in_height, in_width, in_channels

        input_image = np.zeros(shape)

        output_image = make_square(input_image)

        out_height, out_width, out_channels = output_image.shape

        assert out_height == out_width
        assert out_channels == in_channels


def test_even_height_odd_width():
    """
    WHEN: There is an image with odd height, an even width and three channels
    AND: The make_square function is called
    WHAT SHOULD HAPPEN: A square image is returned.
                        Number of channels is preserverd.

    Returns:
        Nothing

    """

    in_height = 4
    in_width = 5

    for in_channels in [1, 3]:

        shape = in_height, in_width, in_channels

        input_image = np.zeros(shape)

        output_image = make_square(input_image)

        out_height, out_width, out_channels = output_image.shape

        assert out_height == out_width
        assert out_channels == in_channels


def test_odd_height_odd_width_square():

    """
    WHEN: There is a square image with odd height, an odd width and three channels
    AND: The make_square function is called
    WHAT SHOULD HAPPEN: The original image is returned.
                        Number of channels is preserverd.

    Returns:
        Nothing

    """

    in_height = 5
    in_width = 5

    for in_channels in [1, 3]:

        shape = in_height, in_width, in_channels

        input_image = np.zeros(shape)

        output_image = make_square(input_image)

        assert np.array_equal(output_image, input_image)


def test_even_height_odd_width_square():

    """
    WHEN: There is a square image with odd height, an odd width and three channels
    AND: The make_square function is called
    WHAT SHOULD HAPPEN: The original image is returned.
                        Number of channels is preserverd.

    Returns:
        Nothing

    """

    in_height = 4
    in_width = 4

    for in_channels in [1, 3]:

        shape = in_height, in_width, in_channels

        input_image = np.zeros(shape)

        output_image = make_square(input_image)

        assert np.array_equal(output_image, input_image)

