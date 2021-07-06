from utils.unicorn_tools import read_mhd_image
from utils.unicorn_tools import dice
from utils.unicorn_tools import original_gt_seg_diff_vis
from utils.unicorn_tools import save_mhd_image

from segmentation_solution import generate_markers
from segmentation_solution import seperate_lungs
from segmentation_solution import get_filtered_lung
from segmentation_solution import get_segmented_lungs

import matplotlib.pyplot as plt
import numpy as np


sample_image_path = "./LungImage/0001.mhd"
sample_mask_path = "./LungImage/0001Mask.mhd"

mhd_image = read_mhd_image(sample_image_path)
mhd_mask = read_mhd_image(sample_mask_path)


dice_all = 0
seg_res = np.ones(mhd_image.shape)
lung_seg_res = np.ones(mhd_image.shape)
seg_diff = np.ones(mhd_image.shape)


for mhd_image_idx in range(mhd_image.shape[0]):

    print(mhd_image_idx, "Lungfilter after closing")
    test_lungfilter_s, lung_s = get_segmented_lungs(mhd_image[mhd_image_idx])

    
    # plt.imshow(test_lungfilter_s, cmap='gray')
    # plt.show()
    #
    # plt.imshow(lung_s, cmap='gray')
    # plt.show()
    # original_gt_seg_diff_vis(mhd_image[mhd_image_idx], mhd_mask[mhd_image_idx], test_lungfilter_s)

    
    seg_res[mhd_image_idx] = test_lungfilter_s
    lung_seg_res[mhd_image_idx] = lung_s
    seg_diff[mhd_image_idx] = (test_lungfilter_s > 0) ^ (mhd_mask[mhd_image_idx] > 0)

   
    dice_s = dice(test_lungfilter_s, mhd_mask[mhd_image_idx])
    # print('similarity for single slice score is {}'.format(dice_s))
    dice_all = dice_all + dice_s

print('Dice similarity score is {}'.format(dice_all/mhd_image.shape[0]))
save_mhd_image(seg_res, "./Result/seg_res.mhd")
save_mhd_image(lung_seg_res, "./Result/lung_seg_res.mhd")
save_mhd_image(seg_diff, "./Result/seg_diff_res.mhd")

