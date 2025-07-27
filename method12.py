import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import umcglib.images as im
from umcglib.utils import tsprint

from expand_lines import dilate_lines, dilate_mask


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


def method_2(
    seg_path, image_path, segmentation_output_path, mask_output_path, image_output_path
):
    tsprint("Reading image...")
    img_s = sitk.ReadImage(seg_path, sitk.sitkFloat32)
    img_n = sitk.GetArrayFromImage(img_s).T

    full_scan_s = sitk.ReadImage(image_path, sitk.sitkFloat32)
    full_scan_n = sitk.GetArrayFromImage(full_scan_s).T

    spacing = img_s.GetSpacing()

    tsprint("Isolating ureter segments..")
    ureter_map = (img_n == 2) * 1
    kidney_map = (img_n == 1) * 1
    bladder_map = (img_n == 3) * 1

    tsprint("Getting ureter components..")
    ureter_components, num_ur_components = ndimage.label(ureter_map)

    tsprint("Getting kidney components..")
    kidney_components, num_ki_components = ndimage.label(kidney_map)

    tsprint("Getting bladder components..")
    bladder_components, num_bl_components = ndimage.label(bladder_map)

    dilated_map = np.zeros_like(img_n)

    tsprint("Dilating ureters..")
    for i in range(1, num_ur_components + 1):
        component = (ureter_components == i) * 1
        dilated_component = dilate_mask(component, 20, spacing)
        dilated_map[dilated_component>0.5] = 2.

    tsprint("Dilating kidneys..")
    for i in range(1, num_ki_components + 1):
        component = (kidney_components == i) * 1
        dilated_component = dilate_mask(component, 10, spacing)
        dilated_map[dilated_component>0.5] = 1.

    tsprint("Dilating kidneys..")
    for i in range(1, num_bl_components + 1):
        component = (bladder_components == i) * 1
        dilated_component = dilate_mask(component, 10, spacing)
        dilated_map[dilated_component>0.5] = 3.

    tsprint("Exporting full segmentation..")
    new_segmentation = dilated_map

    full_mask = (new_segmentation < 0.5) * 1.0
    im.to_sitk(new_segmentation, img_s, save_as=segmentation_output_path)
    im.to_sitk(full_mask, img_s, save_as=mask_output_path)

    full_scan_n[full_mask > 0.5] = -1003
    im.to_sitk(full_scan_n, img_s, save_as=image_output_path)
    tsprint("Done.")



def method_1(
    seg_path, image_path, segmentation_output_path, mask_output_path, image_output_path
):
    tsprint("Reading image...")
    img_s = sitk.ReadImage(seg_path, sitk.sitkFloat32)
    img_n = sitk.GetArrayFromImage(img_s).T

    full_scan_s = sitk.ReadImage(image_path, sitk.sitkFloat32)
    full_scan_n = sitk.GetArrayFromImage(full_scan_s).T

    tsprint("Exporting full segmentation..")
    new_segmentation = img_n

    full_mask = (new_segmentation < 0.5) * 1.0
    im.to_sitk(new_segmentation, img_s, save_as=segmentation_output_path)
    im.to_sitk(full_mask, img_s, save_as=mask_output_path)

    full_scan_n[full_mask > 0.5] = -1003
    im.to_sitk(full_scan_n, img_s, save_as=image_output_path)
    tsprint("Done.")


if __name__ == "__main__":
    import os
    import sys
    from os import path
    from glob import glob

    SEG_DIR = "hema_ct/segmentations/generated/"
    SEG_OUTPUT_DIR_M1 = "hema_ct/final/method1/segmentations/"
    MASK_OUTPUT_DIR_M1 = "hema_ct/final/method1/masks/"
    MASKED_SCAN_OUTPUT_DIR_M1 = "hema_ct/final/method1/masked_scans/"
    SEG_OUTPUT_DIR_M2 = "hema_ct/final/method2/segmentations/"
    MASK_OUTPUT_DIR_M2 = "hema_ct/final/method2/masks/"
    MASKED_SCAN_OUTPUT_DIR_M2 = "hema_ct/final/method2/masked_scans/"

    scan_paths = [l.strip() for l in open("final_scans.txt")]

    current_job_id = int(sys.argv[1])
    num_jobs = int(sys.argv[2])

    paths_per_job = len(scan_paths) // num_jobs + 1
    start_idx = current_job_id * paths_per_job
    end_idx = (current_job_id + 1) * paths_per_job

    tsprint("Processing:", start_idx, "to", end_idx)
    for scan_path in scan_paths[start_idx:end_idx]:
        tsprint("PATH:", scan_path)
        basename = path.basename(scan_path)
        seg_path = path.join(SEG_DIR, basename)

        if not path.exists(seg_path):
            tsprint("Skipping..")
            continue

        tsprint("Processing with METHOD 1..")
        method_1(
            seg_path,
            scan_path,
            SEG_OUTPUT_DIR_M1 + basename,
            MASK_OUTPUT_DIR_M1 + basename,
            MASKED_SCAN_OUTPUT_DIR_M1 + basename,
        )

        tsprint("Processing with METHOD 2..")
        method_2(
            seg_path,
            scan_path,
            SEG_OUTPUT_DIR_M2 + basename,
            MASK_OUTPUT_DIR_M2 + basename,
            MASKED_SCAN_OUTPUT_DIR_M2 + basename,
        )
