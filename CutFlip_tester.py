# This file was created to test our CutFlip data augmentation technique

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import random
import copy

from datasets.drivingstereo_dataset import DrivingStereoDataset

# Load the image
# Update the path below to match the image you want on your system
# This was successfully used to access images on DrivingStereo dataset
image_path = '/monodepth2/assets/test_image.jpg'
image = pil.open(image_path).convert('RGB')

# Apply the Cut_Flip function
augmented_image = DrivingStereoDataset.get_Cut_Flip(image)  # the input image is given to get_Cut_Flip function in DrivingStereoDataset class

# Plot the original and augmented images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image)
ax[0].set_title('Original Image')

ax[1].imshow(augmented_image)
ax[1].set_title('Augmented Image')

# Plotting the images didn't work in SCITAS environment so an option for saving them was added

# Save the original and augmented images
original_image_path = '/monodepth2/assets/original_image.jpg'   # Update the path below to match the location you want on your system
augmented_image_path = '/monodepth2/assets/augmented_image.jpg' # Update the path below to match the location you want on your system

image.save(original_image_path)
augmented_image.save(augmented_image_path)

plt.show()