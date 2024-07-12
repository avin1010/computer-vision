#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
# Read the image and convert to double precision array l = 
plt.imread(r"C:\Users\91866\Downloads\abc.jpeg").astype(float)
# Perform 2-D FFT f1 = 
np.fft.fft2(l)
# Shift zero frequency component to the center f2 = 
np.fft.fftshift(f1)
# Display magnitude of frequency spectrum plt.subplot(2, 2, 1)
plt.imshow(np.abs(f1))
plt.title('Frequency Spectrum')
plt.imshow(np.abs(f2)) 
plt.title('Centered Spectrum')
# Compute log(1 + abs(f2)) f3 = 
np.log(1 + np.abs(f2))
# Display log(1 + abs(f2)) plt.subplot(2, 2, 3) plt.imshow(f3)
plt.title('log(1+abs(f2))')
# Perform 2-D FFT on f1 l_fft = 
fft2(f1)
# Take real part of the result l1 = 
np.real(l_fft)
# Display real part of 2-D FFT 
plt.subplot(2, 2, 4) 
plt.imshow(l1)
plt.title('2-D FFT')
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

# Load the image
l = plt.imread(r"C:\Users\91866\Downloads\abc.jpeg").astype(float)

# Perform 2-D FFT
f1 = np.fft.fft2(l)

# Shift zero frequency component to the center
f2 = np.fft.fftshift(f1)


# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming f1 is your 2-D FFT result
f1 = np.fft.fft2(l)

# Perform FFT shift
f2 = np.fft.fftshift(f1)

# Take 2-D FFT of f1
l_fft = np.fft.fft2(f1)

# Take real part of the result
l1 = np.real(l_fft)

# Display real part of 2-D FFT
plt.subplot(2, 2, 4)
plt.imshow(l1)
plt.title('Real Part of 2-D FFT')
plt.show()


# In[ ]:




