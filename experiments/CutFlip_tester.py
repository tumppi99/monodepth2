import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import copy

def Cut_Flip(image):
    p = random.random()
    if p < 0.5:
        return image
    image_copy = copy.deepcopy(image)
    h, w, c = image_copy.shape

    N = 2     
    h_list = []
    h_interval_list = []   # height interval
    for i in range(N-1):
        h_list.append(random.randint(int(0.2*h), int(0.8*h)))
    h_list.append(h)
    h_list.append(0)  
    h_list.sort()
    h_list_inv = np.array([h]*(N+1))-np.array(h_list)
    for i in range(len(h_list)-1):
        h_interval_list.append(h_list[i+1]-h_list[i])
    for i in range(N):
        image_copy[h_list[i]:h_list[i+1], :, :] = image[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]

    return image_copy

# Load the image
image_path = 'assets/test_image.jpg'
image = np.array(Image.open(image_path))

# Apply the Cut_Flip function
augmented_image = Cut_Flip(image.copy())  # use .copy() to make a copy of the original array

# Plot the original and augmented images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image)
ax[0].set_title('Original Image')

ax[1].imshow(augmented_image)
ax[1].set_title('Augmented Image')

plt.show()