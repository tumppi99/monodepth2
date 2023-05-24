import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import random
import copy

from datasets.drivingstereo_dataset import DrivingStereoDataset

# Load the image
image_path = 'assets/test_image.jpg'
#image = np.array(Image.open(image_path))
image = pil.open(image_path).convert('RGB')

# Apply the Cut_Flip function
augmented_image = DrivingStereoDataset.get_Cut_Flip(image)  # use .copy() to make a copy of the original array

# Plot the original and augmented images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image)
ax[0].set_title('Original Image')

ax[1].imshow(augmented_image)
ax[1].set_title('Augmented Image')

plt.show()