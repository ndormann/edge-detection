import numpy as np


def laplacian_kernel(sigma: float) -> np.ndarray:
    """
    Create discrete Laplacian of Gaussian kernel, clip to ±3σ
    :param sigma: sigma is the std-deviation and refers to spread of laplacian
    """
    width = np.ceil(3 * sigma)
    support_x, support_y = np.mgrid[
        -width : width + 1, -width : width + 1
    ]  # off by one for upper bound

    log_kernel = (
        ((support_x**2 + support_y**2) / (2 * sigma**2) - 1)
        * np.exp(-(support_x**2 + support_y**2) / (2.0 * sigma**2))
        / (np.pi * sigma**4)
    )  # LoG filter
    return log_kernel


def zero_crossing(log: np.ndarray) -> np.ndarray:
    """
    Zero crossing detection
    :param log: image convolved with Laplacian of Gaussian
    """
    crossings = np.zeros_like(log)
    for i in range(log.shape[0]):
        for j in range(log.shape[1]):
            try:
                if log[i][j] == 0:
                    if (
                        (log[i][j - 1] < 0 and log[i][j + 1] > 0)
                        or (log[i][j - 1] < 0 and log[i][j + 1] < 0)
                        or (log[i - 1][j] < 0 and log[i + 1][j] > 0)
                        or (log[i - 1][j] > 0 and log[i + 1][j] < 0)
                    ):
                        crossings[i][j] = 255
                if log[i][j] < 0:
                    if (
                        (log[i][j - 1] > 0)
                        or (log[i][j + 1] > 0)
                        or (log[i - 1][j] > 0)
                        or (log[i + 1][j] > 0)
                    ):
                        crossings[i][j] = 255
            except IndexError:
                pass
    return crossings


def marr_hildreth_detection(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Directional Edge Detection using Marr-Hildreth Algorithm (https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm)
    :param img: input image
    :param sigma: sigma is the std-deviation and refers to spread of gaussian / laplacian
    """
    # create discrete Gaussian kernel, clip to ±3σ
    log_kernel = laplacian_kernel(sigma)
    
    size = log_kernel.shape[0]
    new_shape = img.shape[0] - size + 1, img.shape[1] - size + 1
    log = np.zeros(new_shape, dtype=float)

    for i in range(img.shape[0] - (size - 1)):
        for j in range(img.shape[1] - (size - 1)):
            window = img[i : i + size, j : j + size]
            log[i, j] = np.sum(window * log_kernel)

    edge = zero_crossing(log)
    return edge
