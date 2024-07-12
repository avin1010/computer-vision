#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color  # Corrected import statement
from scipy.stats import pearsonr

# Read the image
i = io.imread(r"C:\Users\91866\Downloads\abc.jpeg")

# Display original image
plt.subplot(2, 2, 1)
plt.imshow(i)
plt.title('Original Image')

# Convert to grayscale
g = color.rgb2gray(i)  # Assign the result to g

# Display grayscale image
plt.subplot(2, 2, 2)
plt.imshow(g, cmap='gray')
plt.title('Gray Image')

# Crop the image
c = g[100:300, 100:300]

# Display cropped image
plt.subplot(2, 2, 3)
plt.imshow(c, cmap='gray')
plt.title('Cropped Image')

# Calculate mean and standard deviation of the cropped image
m = np.mean(c)
s = np.std(c)
print('m:', m)
print('s:', s)

# Generate checkerboard patterns
checkerboard = np.indices((400, 400)).sum(axis=0) % 2

# Create checkerboard images with different thresholds
k = checkerboard > 0.8
k1 = checkerboard > 0.5

# Display checkerboard images
plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(k, cmap='gray')
plt.title('Image1')

plt.subplot(2, 1, 2)
plt.imshow(k1, cmap='gray')
plt.title('Image2')

# Calculate Pearson correlation coefficient between the two images
r, _ = pearsonr(k.flatten(), k1.flatten())
print('r:', r)

plt.show()


# In[ ]:




