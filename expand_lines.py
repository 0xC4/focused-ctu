import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import umcglib.images as im
from umcglib.utils import tsprint, apply_parallel

def sphere_mask(diameter, spacing):
    shape = [s * diameter + 1 for s in spacing]
    center = [s // 2 for s in shape]
    X, Y, Z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist_from_center = np.sqrt(
        ((X - center[0]) * spacing[0]) ** 2
        + ((Y - center[1]) * spacing[1]) ** 2
        + ((Z - center[2]) * spacing[2]) ** 2
    )
    sphere_mask = (dist_from_center <= diameter / 2) * 1.0

    return sphere_mask


def paste_structure(center, full_shape, structure):
    arr = np.zeros(full_shape)
    shape = structure.shape
    w, h, d = shape
    tsprint(f"> Center: {center}; Shape: {full_shape}; Structure: {shape}")
    start_pos = [c - s // 2 for c, s in zip(center, shape)]
    x0, x1 = start_pos[0], start_pos[0] + w
    y0, y1 = start_pos[1], start_pos[1] + h
    z0, z1 = start_pos[2], start_pos[2] + d

    try:
        arr[x0:x1, y0:y1, z0:z1] = structure
    except:
        tsprint("Caught some error.")
    return arr


def dilate_mask(mask, ball_diameter, spacing):
    shape = mask.shape
    line_coords = np.nonzero(mask)
    structure = sphere_mask(ball_diameter, spacing)
    crop_min_x = max(np.amin(line_coords[0]) - ball_diameter // 2 - 1, 0)
    crop_max_x = min(np.amax(line_coords[0]) + ball_diameter // 2 + 1, shape[0])
    crop_min_y = max(np.amin(line_coords[1]) - ball_diameter // 2 - 1, 0)
    crop_max_y = min(np.amax(line_coords[1]) + ball_diameter // 2 + 1, shape[1])
    crop_min_z = max(np.amin(line_coords[2]) - ball_diameter // 2 - 1, 0)
    crop_max_z = min(np.amax(line_coords[2]) + ball_diameter // 2 + 1, shape[2])

    cropped_mask = mask[
        crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z
    ]
    dilated_crop = ndimage.binary_dilation(cropped_mask, structure)
    mask[crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z] = dilated_crop
    return mask


def dilate_lines(line_mask, diameter, spacing):

    shape = line_mask.shape
    line_coords = np.nonzero(line_mask)
    sphere_4cm = sphere_mask(diameter, spacing)
    sphere_w, sphere_h, sphere_d = sphere_4cm.shape

    crop_min_x = max(np.amin(line_coords[0]) - sphere_w // 2 - 1, 0)
    crop_max_x = min(np.amax(line_coords[0]) + sphere_w // 2 + 1, shape[0])
    crop_min_y = max(np.amin(line_coords[1]) - sphere_h // 2 - 1, 0)
    crop_max_y = min(np.amax(line_coords[1]) + sphere_h // 2 + 1, shape[1])
    crop_min_z = max(np.amin(line_coords[2]) - sphere_d // 2 - 1, 0)
    crop_max_z = min(np.amax(line_coords[2]) + sphere_d // 2 + 1, shape[2])

    cropped_seg = line_mask[
        crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z
    ]
    cropped_shape = cropped_seg.shape
    tsprint("Cropped seg to", cropped_shape)
    # im.to_sitk(cropped_seg, save_as="tmp/cropped_seg.nii.gz")

    num_coords = len(line_coords[0])
    tsprint(num_coords)

    coords = [
        (
            line_coords[0][i] - crop_min_x,
            line_coords[1][i] - crop_min_y,
            line_coords[2][i] - crop_min_z,
        )
        for i in range(num_coords)
    ]

    buffer_size = 24
    buffer_idx = 0

    dilated_crop_n = np.zeros_like(cropped_seg)
    while buffer_idx * buffer_size < len(coords):
        start_idx = buffer_idx * buffer_size
        end_idx = (buffer_idx + 1) * buffer_size
        tsprint(f"Processing {start_idx} to {end_idx}..")

        buffer_coords = coords[start_idx:end_idx]
        buffer_masks = apply_parallel(
            buffer_coords,
            paste_structure,
            full_shape=cropped_shape,
            structure=sphere_4cm,
            num_workers=8,
        )
        buffer_idx += 1

        dilated_crop_n = np.amax([dilated_crop_n] + buffer_masks, 0)

    full_dilated_mask_n = np.zeros_like(line_mask)
    full_dilated_mask_n[
        crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z
    ] = dilated_crop_n

    return full_dilated_mask_n


full_seg_s = sitk.ReadImage("tmp/full_segmentation.nii.gz")
full_seg_n = sitk.GetArrayFromImage(full_seg_s).T

kidney_mask = (full_seg_n == 1) * 1.
im.to_sitk(kidney_mask, ref_img=full_seg_s, save_as="tmp/kidney_undilated.nii.gz")
dilated_kidney_mask = dilate_mask(
    kidney_mask, ball_diameter=10, spacing=full_seg_s.GetSpacing()
)
im.to_sitk(dilated_kidney_mask, ref_img=full_seg_s, save_as="tmp/kidney_dilated.nii.gz")
tsprint("Done")
# spacing = full_seg_s.GetSpacing()

# full_dilated_mask_n = dilate_lines((full_seg_n == 4) * 1.0, 40, spacing)
# full_dilated_mask_s = sitk.GetImageFromArray(full_dilated_mask_n.T)
# sitk.WriteImage(full_dilated_mask_s, "tmp/full_dilated_mask.nii.gz")
# tsprint("Done.")
