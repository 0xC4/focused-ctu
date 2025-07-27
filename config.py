from os import path

import numpy as np
import SimpleITK as sitk

from helpers import *

#############################
##         General         ##
#############################
PROJECT_NAME = "hema_ct"
IMAGE_DIR = "img_data/"
WORK_DIR = "work_dir/"
SEGMENTATION_DIR = path.join(WORK_DIR, "segmentations/")
MANUAL_DIR = "segmentations/"
GENERATED_DIR = path.join(SEGMENTATION_DIR, "generated/")
MASKED_IMAGE_DIR = path.join(SEGMENTATION_DIR, "masked/")
RESAMPLED_SCAN_DIR = path.join(SEGMENTATION_DIR, "resampled_scans/")
GENERATED_TMP_ZIP = path.join(SEGMENTATION_DIR, "generated_tmp.zip")
MASKED_IMAGE_TMP_ZIP = path.join(SEGMENTATION_DIR, "masked_tmp.zip")
OUTPUT_DIR = path.join(WORK_DIR, "output")
SAMPLE_DIR = path.join(WORK_DIR, "samples")
LOG_DIR = path.join(WORK_DIR, "logs")
MODEL_DIR = path.join(WORK_DIR, "models")

#############################
##      Deep learning      ##
#############################
DL_SPACING = [1., 1., 2.]       # Voxel spacing used for deep learning
WINDOW_SIZE = (160, 160, 48)    # Crop size in voxels
BATCH_SIZE = 5                  # Training batch size
LEARNING_RATE = 1e-4            # Learning rate
VALID_NUM_CROPS = 50            # Number of crops used for validation
REPLACEMENT_VALUE = -1000       # Value used to mask unwanted organs
MAX_EPOCHS = 5000                # Maximum number of training epochs
EARLY_STOPPING = 50             # Automatically stop training after X epochs
                                # (None to disable)
MODEL_SEL_VAR = "val_loss"      # Variable to track for determining best model
MODEL_SEL_MODE = "min"          # Whether to minimize or maximize the score

# Calculate the field of view of a single crop in millimeters
FIELD_OF_VIEW = [sp * sz for sp, sz in zip(DL_SPACING, WINDOW_SIZE)]

############################
##         Classes        ##
############################
CLASSES = { 
    0 : "background",
    1 : "nier",
    2 : "ureteren",
    3 : "blaas",
}
NUM_CLASSES = len(CLASSES)

#############################
## Preprocessing function ##
#############################
def preprocess(img, seg = None, 
    dtype="sitk", 
    resample_spacing=None,
    resample_min_shape=None,
    normalization="znorm"):

    if resample_spacing is not None:
        img = resample(img, 
            min_shape=resample_min_shape, 
            method=sitk.sitkLinear,
            new_spacing=resample_spacing)
        if seg is not None:
            seg = resample_to_reference(seg, img)
    img_n = sitk.GetArrayFromImage(img).T
    
    if seg is not None:
        seg_n = sitk.GetArrayFromImage(seg).T

    if normalization == "znorm":
        img_n = img_n - np.mean(img_n)
        img_n = img_n / np.std(img_n)
    elif normalization == "window":
        img_n = np.clip(img_n, -300, 300) / 300
    
    if dtype == "numpy":
        if seg is not None:
            return img_n, seg_n
        return img_n, None

    # Restore SITK parameters
    img_s = sitk.GetImageFromArray(img_n.T)
    img_s.CopyInformation(img)

    if seg is not None:
        seg_s = sitk.GetImageFromArray(seg_n.T)
        seg_s.CopyInformation(seg)

        return img_s, seg_s
    return img_s, None

def get_mask(segmentation):
    """Function to create the mask used to blind the image, based on the 
    generated segmentation.
    """
    mask = (segmentation > 0) * 1.
    return mask
