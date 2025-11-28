# Image Compression Using K-Means Clustering
## Overview

This repository contains an educational project demonstrating how to compress an image by reducing its color space using the K-Means clustering algorithm.
The goal of this project is to illustrate how clustering can be applied to image data and how color quantization can simplify an image while retaining its essential structure.

## How the Program Works

### The program implements the following steps:

1. Loads and displays an input image.
2. Converts the image from a 3D array (height, width, 3) into a 2D array (num_pixels, 3), where each row represents one RGB pixel.
3. Initializes K random centroids selected from the pixel dataset.
4. Assigns each pixel to the nearest centroid and updates centroid positions for a fixed number of iterations.
5. Reconstructs a compressed image by replacing each pixel with the RGB value of its assigned centroid.
6. Displays and saves the resulting compressed image.
7. Shows the original and compressed versions side by side for comparison.

## About the K-Means Algorithm

K-Means is an iterative clustering method used to partition data into K groups based on similarity.

### In this project:

- Each pixel is treated as a point in RGB color space.
- The algorithm groups similar colors into clusters.
- The centroid of each cluster becomes one of the limited colors in the compressed image.
- The final image contains only K unique colors, representing a reduced color palette.

This process is also known as color quantization.

## Results

The output image contains fewer unique colors, depending on the chosen value of K.
Smaller values of K produce stronger compression with more visible color loss, while larger values preserve image detail more effectively.

## Conclusion

This educational project demonstrates how K-Means clustering can be applied to image processing tasks, specifically to reduce the number of colors in an image.
It provides a practical example of reshaping image data, performing clustering, reconstructing a compressed image, and comparing results before and after compression.