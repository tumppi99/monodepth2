import numpy as np
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import PIL.Image as pil
import random
import copy

from datasets.drivingstereo_dataset import DrivingStereoDataset

# Load the image
image_path = 'monodepth2/assets/test_image.jpg'
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

# Save the original and augmented images
original_image_path = 'monodepth2/assets/original_image.jpg'
augmented_image_path = 'monodepth2/assets/augmented_image.jpg'

image.save(original_image_path)
augmented_image.save(augmented_image_path)

plt.show()