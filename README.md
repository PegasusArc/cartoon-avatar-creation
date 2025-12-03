# Avatar Creation

Transform portrait photos into cartoon avatars using OpenCV.

## Features

- K-means color quantization (9 colors)
- Adaptive threshold edge detection  
- Bilateral filtering for smoothing
- 50-100ms processing time (CPU only)
- No GPU or training required
- Minimal dependencies

## Installation

pip install opencv-python numpy

## Usage

import cv2
import numpy as np

def reduce_colors(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()].reshape(img.shape)
    return result

def get_edges(img, line_size, blur_val):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_val)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, line_size, blur_val)
    return edges

# Load image
img = cv2.imread("portrait.jpg")

# Step 1: Extract edges
edges = get_edges(img, line_size=7, blur_val=9)

# Step 2: Reduce colors
img = reduce_colors(img, k=9)

# Step 3: Smooth with bilateral filter
blurred = cv2.bilateralFilter(img, d=3, sigmaColor=200, sigmaSpace=200)

# Step 4: Apply edge mask
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

# Save result
cv2.imwrite("cartoon_avatar.jpg", cartoon)

## How It Works

1. **Edge Detection**: Converts image to grayscale, applies median blur, 
   then uses adaptive thresholding to extract bold cartoon outlines

2. **Color Quantization**: K-means clustering reduces the image to 9 distinct 
   colors for flat cartoon appearance

3. **Bilateral Filtering**: Edge-preserving smoothing that maintains

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_val)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, line_size, blur

