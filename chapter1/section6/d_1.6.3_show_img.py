#!env python3
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./dataset/psq.jpeg')
plt.imshow(img)
plt.show()
