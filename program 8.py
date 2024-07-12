#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Read the image
I = cv2.imread(r"C:\Users\91866\Downloads\abc.jpeg")

# Convert to grayscale
K = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Add salt and pepper noise
J = K.copy()
noise = np.random.choice([0, 255], K.shape, p=[0.95, 0.05])
J[noise == 255] = 255
J[noise == 0] = 0

# Apply median filters
f = median_filter(J, size=(3, 3))
f1 = median_filter(J, size=(10, 10))

# Display results
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Grayscale image
plt.subplot(3, 2, 2)
plt.imshow(K, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

# Noisy image
plt.subplot(3, 2, 3)
plt.imshow(J, cmap='gray')
plt.title('Noise added Image')
plt.axis('off')

# Median filtered images
plt.subplot(3, 2, 4)
plt.imshow(f, cmap='gray')
plt.title('3x3 Median Filter')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(f1, cmap='gray')
plt.title('10x10 Median Filter')
plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




