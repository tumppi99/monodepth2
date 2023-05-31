# Created for DrivingStereo dataset implementation
# Same idea as in kitti_dataset file by the authors

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import random
import copy

#from kitti_utils import generate_depth_map, Commented out because DrivigStereo has ready-made depth maps so no need to generate one
from .mono_dataset import MonoDataset

class DrivingStereoDataset(MonoDataset):

    def __init__(self, *args, **kwargs):
        super(DrivingStereoDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        # Camera intrinsics
        # 15mm images have different focals
        self.c_u = 4.556890e+2
        self.c_v = 1.976634e+2
        self.f_u = 1.003556e+3
        self.f_v = 1.003556e+3

        image_width = 864
        image_height = 384
        
        # Defining the intrinsics matrix parameters based on the values presented above
        self.K = np.array([[self.f_u / image_width, 0, 0.5, 0],
                           [0, self.f_v / image_height, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (864, 384)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    # Produce the path to find the correct camera images from DrivingStereo dataset in SCITAS
    def get_image_path(self, folder):
        image_path = os.path.join(self.data_path,
                                    'train-left-image',
                                    folder + '.jpg')
        # In our case "folder" contains names of picture frames without the filetype at the end (".jpg" or ".png")
        # That is why the type is added at the end. Also, we specify that we want left training images
        # Reading of the folders and files were dificult to adapt due to different structures of KITTI and DrivingStereo datasets
        with open('/home/herlevi/monodepth2/debugs/debug_get_image_path.txt', 'w') as f:
            # Check that the folder path is correct
            f.write(image_path)
            
        return image_path

    # Produce the path to find the correct depth map from DrivingStereo dataset in SCITAS
    def get_depth(self, folder):
        depth_path = os.path.join(self.data_path,
                                  'train-depth-map',
                                  folder + '.png')
        # In our case "folder" contains names of picture frames without the filetype at the end (".jpg" or ".png")
        # That is why the type is added at the end. Also, we specify that we want depth map images
        # Reading of the folders and files were dificult to adapt due to different structures of KITTI and DrivingStereo datasets
        with open('/home/herlevi/monodepth2/debugs/debug_get_image_path_depth.txt', 'w') as f:
            # Check that the folder path is correct
            f.write(depth_path)
        
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        return depth_gt        

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    '''
    No need for creating depth maps

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    '''
    
    # The CutFlip operation is executed here
    def get_Cut_Flip(image):
        # CutFlip is not done to every frame due to the random factor
        p = np.random.random()
        if p < 0.5:
            return image

        image_copy = image.copy()
        image_array = np.array(image_copy)

        h, w, c = image_array.shape

        # The vertical location of the horizontal cut is also randomized
        h_cut = np.random.randint(int(0.2 * h), int(0.8 * h))

        upper_part = image_array[:h_cut, :, :]
        lower_part = image_array[h_cut:, :, :]

        # The cutted parts are flipped so that the upper part goes down and vice versa
        flipped_image_array = np.concatenate((lower_part, upper_part), axis=0)

        flipped_image = pil.fromarray(flipped_image_array)
        
        return flipped_image