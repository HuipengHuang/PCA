import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

image_path = r"C:\Users\27859\Desktop\PyCharm 2024.1.3\jbr\bin\E\Users\27859\PycharmProjects\optim_project\data\yaleB\subject01.glasses"
image = imread(image_path)
print("Image shape:", image.shape)  # Likely (1, 243, 320) or (243, 320)
plt.imshow(image.reshape(243, 320, 1), cmap='gray')
plt.show()