import numpy as np


def symmetric_convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve image with kernel
    :param img: input image
    :param kernel: kernel to convolve with
    """
    tmp = np.zeros_like(img, dtype=float)
    gauss = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        tmp[i, :] = np.convolve(img[i, :], kernel, mode="same")
    for j in range(img.shape[1]):
        gauss[:, j] = np.convolve(tmp[:, j], kernel, mode="same")
    return gauss


def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Create discrete Gaussian kernel, clip to ±3σ
    :param sigma: sigma is the std-deviation and refers to spread of gaussian
    """
    width = np.ceil(3 * sigma)
    size = int(2 * width + 1)
    support = np.arange(-width, width + 1)  # off by one for upper bound
    gauss_kernel = np.exp(-(support**2) / (2.0 * sigma**2)) / (
        sigma * np.sqrt(2 * np.pi)
    )
    return gauss_kernel


def sobel_kernel() -> np.ndarray:
    """
    Create discrete Sobel kernel
    """
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def non_maximum_suppression(magnitude: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Non maximum suppression
    :param magnitude: magnitude of gradient
    :param theta: angle of gradient (degree)
    """

    theta[theta < 0] += 180
    nms = np.copy(magnitude)
    for i in range(theta.shape[0] - 1):
        for j in range(theta.shape[1] - 1):
            try:
                if theta[i, j] <= 22.5 or theta[i, j] > 157.5:
                    if (magnitude[i, j] <= magnitude[i - 1, j]) or (
                        magnitude[i, j] <= magnitude[i + 1, j]
                    ):
                        nms[i, j] = 0
                elif theta[i, j] > 22.5 and theta[i, j] <= 67.5:
                    if (magnitude[i, j] <= magnitude[i - 1, j - 1]) or (
                        magnitude[i, j] <= magnitude[i + 1, j + 1]
                    ):
                        nms[i, j] = 0
                elif theta[i, j] > 67.5 and theta[i, j] <= 112.5:
                    if (magnitude[i, j] <= magnitude[i + 1, j + 1]) or (
                        magnitude[i, j] <= magnitude[i - 1, j - 1]
                    ):
                        nms[i, j] = 0
                elif theta[i, j] > 112.5 and theta[i, j] <= 157.5:
                    if (magnitude[i, j] <= magnitude[i + 1, j - 1]) or (
                        magnitude[i, j] <= magnitude[i - 1, j + 1]
                    ):
                        nms[i, j] = 0
            except IndexError:
                pass
    return nms


def thresholding(
    img: np.ndarray, lowThreshold: float, highThreshold: float
) -> np.ndarray:
    """
    Thresholding
    :param img: input image
    :param lowThreshold: low threshold used to identify weak edges
    :param highThreshold: high threshold used to identify strong edges
    :return: edge image with weak and strong edges
    """
    weak = 50
    strong = 255
    highThreshold = int(img.max() * highThreshold)
    lowThreshold = int(img.max() * lowThreshold)
    thresh = np.zeros_like(img, dtype=np.int32)
    thresh[img >= highThreshold] = strong
    thresh[(img > lowThreshold) & (img < highThreshold)] = weak
    return thresh, weak, strong


def hysteresis(
    img: np.ndarray, weak: int, strong: int, blob_size: int = 2
) -> np.ndarray:
    """
    Hysteresis
    :param img: input image
    :param weak: weak edges
    :param strong: strong edges
    """
    hys = np.copy(img)
    for i in range(hys.shape[0] - 1):
        for j in range(hys.shape[1] - 1):
            if hys[i, j] == weak:
                hys[i, j] = 0
                for x in range(i - blob_size, i + blob_size + 1):
                    for y in range(j - blob_size, j + blob_size + 1):
                        try:
                            if hys[x, y] == strong:
                                hys[i, j] = strong
                                break
                        except IndexError:
                            pass
    return hys


def canny_detection(
    img: np.ndarray, sigma: float, weak_threshold: float, strong_threshold: float
) -> np.ndarray:
    """
    Directional Edge Detection using Canny Algorithm (https://en.wikipedia.org/wiki/Canny_edge_detector)
    :param img: input image
    :param sigma: sigma is the std-deviation and refers to spread of gaussian
    :param threshold: threshold used to identify edges
    """
    # create discrete Gaussian kernel, clip to ±3σ
    gauss_kernel = gaussian_kernel(sigma)
    gauss = symmetric_convolve(img, gauss_kernel)

    # differentation with sobel kernel
    diff_kernel = sobel_kernel()
    size = diff_kernel.shape[0]
    gx = np.zeros_like(gauss, dtype=float)
    gy = np.zeros_like(gauss, dtype=float)
    for i in range(gauss.shape[0] - (size - 1)):
        for j in range(gauss.shape[1] - (size - 1)):
            window = gauss[i : i + size, j : j + size]
            gx[i, j], gy[i, j] = (
                np.sum(window * diff_kernel.T),
                np.sum(window * diff_kernel),
            )

    magnitude = np.hypot(gx, gy)
    magnitude = magnitude / magnitude.max() * 255
    theta = np.rad2deg(np.arctan2(gx, gy))
    nms = non_maximum_suppression(magnitude, theta)

    thresh, weak, strong = thresholding(nms, weak_threshold, strong_threshold)

    hys = hysteresis(thresh, weak, strong)
    return hys
