#test for drivingstereo implementation

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import random
import copy

#from kitti_utils import generate_depth_map    depth mappii ei ainakaa tartte     drivingstereo_utils ?
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
        self.K = np.array([[2.34, 0, 0.5, 0],
                           [0, 5.15, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32) #ei tietoa parametreista

        self.full_res_shape = (881, 400)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3} #tarviiko?

        '''
        # Camera intrinsics
        # 15mm images have different focals
        self.c_u = 4.556890e+2
        self.c_v = 1.976634e+2
        self.f_u = 1.003556e+3
        self.f_v = 1.003556e+3
        '''
    
    def get_image_path(self, folder):
        image_path = os.path.join(self.data_path, 'train-left-image', folder)
        with open('/home/herlevi/monodepth2/debugs/debug_get_image_path.txt', 'w') as f:
            f.write(image_path)
        return image_path


    def get_depth(self, folder):
        image_path = os.path.join(self.data_path, 'train-depth-map', folder)
        with open('/home/herlevi/monodepth2/debugs/debug_get_image_path_depth.txt', 'w') as f:
            f.write(image_path)
        return image_path
        

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    
    def get_Cut_Flip(image):
        p = np.random.random()
        if p < 0.5:
            return image

        image_copy = image.copy()
        image_array = np.array(image_copy)

        h, w, c = image_array.shape

        h_cut = np.random.randint(int(0.2 * h), int(0.8 * h))

        upper_part = image_array[:h_cut, :, :]
        lower_part = image_array[h_cut:, :, :]

        flipped_image_array = np.concatenate((lower_part, upper_part), axis=0)

        flipped_image = pil.fromarray(flipped_image_array)
        return flipped_image
    

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