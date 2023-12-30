Python Implementations of Standard Edge Detection Algorithms
===================

## Requirements
- NumPy
- Matplotlib
- Pillow

## Sources
- Image from Davidwkennedy - http://en.wikipedia.org/wiki/File:Bikesgray.jpg - [License](https://creativecommons.org/licenses/by-sa/3.0/deed.en)
- [Canny Edge Detector Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Marr-Hildreth Edge Detector Wikipedia](https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm)

## Canny Algorithm [Code](https://github.com/ndormann/edge-detection/blob/main/canny.py)
- Smooth with Gaussian filter
- Differentiate via Sobel kernel
- Apply non-maximum suppression
- Apply double threshold to find weak and strong edges
- Edge tracking by hysteresis

## Marr-Hildreth Algorithm [Code](https://github.com/ndormann/edge-detection/blob/main/marrHildreth.py)
- Convolve with Laplacian of Gaussian
- Find Zero-Crossings

## Results
### Original
![Original](https://github.com/ndormann/edge-detection/blob/main/Bikesgray.jpg)

### Canny
![Canny Edges](https://github.com/ndormann/edge-detection/blob/main/canny.png)

### Marr-Hildreth
![Marr-Hildreth Edges](https://github.com/ndormann/edge-detection/blob/main/marr-hildreth.png)
