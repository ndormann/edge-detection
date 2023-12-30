#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np

import canny
import marrHildreth


def main():
    oparser = argparse.ArgumentParser(
        description="Edge detection with Canny and Hildreth/Marr"
    )
    oparser.add_argument(
        "--input",
        dest="input_image",
        default="Bikesgray.jpg",
        required=False,
        help="Path containing the image",
    )
    oparser.add_argument(
        "--sigma",
        dest="sigma",
        default=3.0,
        required=False,
        help="Sigma for Gaussian",
        type=float,
    )
    oparser.add_argument(
        "--wth",
        dest="weak_threshold",
        default=0.05,
        required=False,
        help="Weak Threshold for edges",
        type=float,
    )
    oparser.add_argument(
        "--sth",
        dest="strong_threshold",
        default=0.1,
        required=False,
        help="Storng Threshold for edges",
        type=float,
    )
    options = oparser.parse_args()

    with Image.open(options.input_image) as im:
        im = im.convert("L")
        img = np.array(im)

        res_canny = canny.canny_detection(
            img, options.sigma, options.weak_threshold, options.strong_threshold
        )
        res_marr = marrHildreth.marr_hildreth_detection(img, options.sigma)

        Image.fromarray(np.array(res_canny, dtype=np.uint8), mode="L").save("canny.png")
        Image.fromarray(np.array(res_marr, dtype=np.uint8), mode="L").save("marr-hildreth.png")

        plt.subplot(121)
        plt.title("Canny")
        plt.imshow(res_canny, cmap="gray")
        plt.subplot(122)
        plt.title("Marr-Hildreth")
        plt.imshow(res_marr, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
