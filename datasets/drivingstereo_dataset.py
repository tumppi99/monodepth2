#test for drivingstereo implementation

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map #drivingstereo_utils ?
from .mono_dataset import MonoDataset


class DrivingStereoDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DrivingStereoDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32) #ei tietoa parametreista

        self.full_res_shape = (881, 400)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3} #tarviiko?

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    

'''
Chat GPT vinkkejä

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

# Update the import statement to match the file containing the necessary functions for the DrivingStereo dataset.
from drivingstereo_utils import generate_depth_map
from .mono_dataset import MonoDataset


class DrivingStereoDataset(MonoDataset):
    """Superclass for different types of DrivingStereo dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DrivingStereoDataset, self).__init__(*args, **kwargs)

        # Update the intrinsics matrix to match the specifications of the DrivingStereo dataset.
        # Replace the values below with the correct intrinsics matrix for DrivingStereo.
        self.K = np.array([[focal_length_x, 0, principal_point_x, 0],
                           [0, focal_length_y, principal_point_y, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (image_width, image_height)  # Update with the actual image resolution of DrivingStereo.
        self.side_map = {"left": 2, "right": 3}  # Update the mapping for the left and right camera sides.

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
'''

'''
def Cut_Flip(self, image, depth):

        p = random.random()
        if p < 0.5:
            return image, depth
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        h, w, c = image.shape

        N = 2     
        h_list = []
        h_interval_list = []   # hight interval
        for i in range(N-1):
            h_list.append(random.randint(int(0.2*h), int(0.8*h)))
        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h]*(N+1))-np.array(h_list)
        for i in range(len(h_list)-1):
            h_interval_list.append(h_list[i+1]-h_list[i])
        for i in range(N):
            image[h_list[i]:h_list[i+1], :, :] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
            depth[h_list[i]:h_list[i+1], :, :] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]

        return image, depth
'''