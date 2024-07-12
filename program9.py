#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os

# Define function to read image safely
def safe_imread(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return cv2.imread(filename)

# Define the Laplacian filter
def laplacian_filter(img, alpha=0.05):
    kernel = np.array([[0, 1, 0], [1, -4 + alpha, 1], [0, 1, 0]])
    return convolve(img, kernel)

# Define Prewitt filters for horizontal and vertical edge detection
def prewitt_filter(img):
    kernel_h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    gx = convolve(img, kernel_h)
    gy = convolve(img, kernel_v)
    return np.sqrt(gx**2 + gy**2)

# Define Roberts filters for horizontal and vertical edge detection
def roberts_filter(img):
    kernel_h = np.array([[1, 0], [0, -1]])
    kernel_v = np.array([[0, 1], [-1, 0]])
    gx = convolve(img, kernel_h)
    gy = convolve(img, kernel_v)
    return np.sqrt(gx**2 + gy**2)

# Define Sobel filters for horizontal and vertical edge detection
def sobel_filter(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)

# Main script
try:
    # Read the image
    i = safe_imread(r"C:\Users\91866\Downloads\apple1.jpeg")

    # Display the original image
    plt.subplot(4, 2, 1)
    plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Convert to grayscale
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image
    plt.subplot(4, 2, 2)
    plt.imshow(g, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')

    # Apply Laplacian filter
    f_laplacian = laplacian_filter(g, alpha=0.05)

    # Display the Laplacian filtered image
    plt.subplot(4, 2, 3)
    plt.imshow(f_laplacian, cmap='gray')
    plt.title('Laplacian')
    plt.axis('off')

    # Apply Prewitt filter
    f_prewitt = prewitt_filter(g)

    # Display the Prewitt edge detected image
    plt.subplot(4, 2, 4)
    plt.imshow(f_prewitt, cmap='gray')
    plt.title('Prewitt')
    plt.axis('off')

    # Apply Roberts filter
    f_roberts = roberts_filter(g)

    # Display the Roberts edge detected image
    plt.subplot(4, 2, 5)
    plt.imshow(f_roberts, cmap='gray')
    plt.title('Roberts')
    plt.axis('off')

    # Apply Sobel filter
    f_sobel = sobel_filter(g)

    # Display the Sobel edge detected image
    plt.subplot(4, 2, 6)
    plt.imshow(f_sobel, cmap='gray')
    plt.title('Sobel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(e)


# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define function to read image safely
def safe_imread(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return cv2.imread(filename)

# Main script
try:
    # Read the image
    i = safe_imread(r"C:\Users\91866\Downloads\apple1.jpeg")

    # Display the original image
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Convert to grayscale
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image
    plt.subplot(3, 2, 2)
    plt.imshow(g, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')

    # Apply Sobel filter for horizontal edge detection
    sobelx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    
    # Display the horizontal Sobel edge detected image
    plt.subplot(3, 2, 3)
    plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel Horizontal')
    plt.axis('off')

    # Apply Sobel filter for vertical edge detection
    sobely = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    
    # Display the vertical Sobel edge detected image
    plt.subplot(3, 2, 4)
    plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Vertical')
    plt.axis('off')

    # Combine Sobel x and y outputs to get the magnitude
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)

    # Display the combined Sobel edge detected image
    plt.subplot(3, 2, 5)
    plt.imshow(sobel_mag, cmap='gray')
    plt.title('Sobel Magnitude')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(e)


# In[ ]:




