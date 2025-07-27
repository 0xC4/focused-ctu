import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import umcglib.images as im
from umcglib.utils import tsprint

from expand_lines import dilate_lines, dilate_mask


def shortest_distance(component1: np.ndarray, component2: np.ndarray, max_samples=1500):
    # Finds the shortest distance between two components
    coords1 = np.nonzero(component1)
    coords2 = np.nonzero(component2)

    num_coords1 = len(coords1[0])
    num_coords2 = len(coords2[0])

    indexes1 = list(range(num_coords1))
    indexes2 = list(range(num_coords2))
    np.random.shuffle(indexes1)
    np.random.shuffle(indexes2)

    if max_samples:
        indexes1 = indexes1[:max_samples]
        indexes2 = indexes2[:max_samples]

    shortest_dist = 999999999
    shortest_coord1 = None
    shortest_coord2 = None
    for idx1 in indexes1:
        x1, y1, z1 = coords1[0][idx1], coords1[1][idx1], coords1[2][idx1]
        for idx2 in indexes2:
            x2, y2, z2 = coords2[0][idx2], coords2[1][idx2], coords2[2][idx2]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

            if distance < shortest_dist:
                # tsprint(f"Found shorter distance: {distance} (prev {shortest_dist})")
                shortest_dist = distance
                shortest_coord1 = x1, y1, z1
                shortest_coord2 = x2, y2, z2

    return shortest_coord1, shortest_coord2, distance


def draw_line(arr, x0, y0, z0, x1, y1, z1):
    out_lines = np.zeros_like(arr)

    # Loop over rounded X Y and Z intervals between target
    x_distance = abs(x0 - x1)
    y_distance = abs(y0 - y1)
    z_distance = abs(z0 - z1)

    if x0 < x1:
        min_x, max_x = x0, x1
        y_start, y_end = y0, y1
        z_start, z_end = z0, z1
    else:
        min_x, max_x = x1, x0
        y_start, y_end = y1, y0
        z_start, z_end = z1, z0

    for x in range(round(min_x), round(max_x)):
        proportion = (x - min_x) / x_distance
        interpolated_y = y_start + proportion * (y_end - y_start)
        interpolated_z = z_start + proportion * (z_end - z_start)
        pixel_y, pixel_z = int(round(interpolated_y)), int(round(interpolated_z))
        out_lines[x, pixel_y, pixel_z] = 1.0

    if y0 < y1:
        min_y, max_y = y0, y1
        x_start, x_end = x0, x1
        z_start, z_end = z0, z1
    else:
        min_y, max_y = y1, y0
        x_start, x_end = x1, x0
        z_start, z_end = z1, z0

    for y in range(round(min_y), round(max_y)):
        proportion = (y - min_y) / y_distance
        interpolated_x = x_start + proportion * (x_end - x_start)
        interpolated_z = z_start + proportion * (z_end - z_start)
        pixel_x, pixel_z = int(round(interpolated_x)), int(round(interpolated_z))
        out_lines[pixel_x, y, pixel_z] = 1.0

    if z0 < z1:
        min_z, max_z = z0, z1
        x_start, x_end = x0, x1
        y_start, y_end = y0, y1
    else:
        min_z, max_z = z1, z0
        x_start, x_end = x1, x0
        y_start, y_end = y1, y0

    for z in range(round(min_z), round(max_z)):
        proportion = (z - min_z) / z_distance
        interpolated_x = x_start + proportion * (x_end - x_start)
        interpolated_y = y_start + proportion * (y_end - y_start)
        pixel_x, pixel_y = int(round(interpolated_x)), int(round(interpolated_y))
        out_lines[pixel_x, pixel_y, z] = 1.0

    return out_lines


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


def method_3(
    seg_path, image_path, segmentation_output_path, mask_output_path, image_output_path
):
    tsprint("Reading image...")
    # seg_path = "/scratch/p286425/hema_ct/segmentations/generated/train_133.nii.gz"
    img_s = sitk.ReadImage(seg_path, sitk.sitkFloat32)
    img_n = sitk.GetArrayFromImage(img_s).T

    full_scan_s = sitk.ReadImage(image_path, sitk.sitkFloat32)
    full_scan_n = sitk.GetArrayFromImage(full_scan_s).T

    spacing = img_s.GetSpacing()
    w, h, d = img_n.shape

    x_center = w // 2

    # sphere_mask_4cm = sphere_mask(40, spacing)[..., ::8]
    # im.to_sitk(sphere_mask_4cm, spacing=spacing, save_as="tmp/sphere_mask.nii.gz")

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

    # tsprint("Writing ureter components to file..")
    # im.to_sitk(ureter_components, save_as="tmp/ureter_components.nii.gz", ref_img=img_s)
    # tsprint("Done.")

    tsprint("Separating ureter components into left and right..")
    components_by_side = {"left": [], "right": []}
    for i in range(1, num_ur_components + 1):
        component = (ureter_components == i) * 1
        coordinates = np.nonzero(component)
        coordinates = [np.mean(c) for c in coordinates]

        x_coord = coordinates[0]
        if x_coord > x_center:
            side = "left"  # medical orientation
        else:
            side = "right"
        tsprint(f"Component: {i}, Avg X coord: {x_coord}, Side: {side}", coordinates)
        components_by_side[side].append(component)

    tsprint("Separating kidney components into left and right..")
    kidneys_by_side = {"left": [], "right": []}
    for i in range(1, num_ki_components + 1):
        component = (kidney_components == i) * 1
        coordinates = np.nonzero(component)
        coordinates = [np.mean(c) for c in coordinates]

        x_coord = coordinates[0]
        if x_coord > x_center:
            side = "left"  # medical orientation
        else:
            side = "right"
        tsprint(f"Component: {i}, Avg X coord: {x_coord}, Side: {side}", coordinates)
        kidneys_by_side[side].append(component)

    tsprint("Enumerating bladder components..")
    bladder_comps = [
        (bladder_components == i) * 1 for i in range(1, num_bl_components + 1)
    ]

    lines = np.zeros_like(ureter_components)
    merged_components = []
    for side in ["left", "right"]:
        comps = components_by_side[side]
        if len(comps) < 1:
            tsprint(f"No ureter components found for {side} side")
            continue
        main_component = comps.pop()
        merged_components.append(main_component)

        side_lines = np.zeros_like(lines)
        while len(comps) > 0:
            tsprint("Finding nearest component to merge with..")
            best_candidate_idx = None
            best_distance = 9999999
            best_coord_main = None
            best_coord_candidate = None
            for candidate_idx, merge_candidate in enumerate(comps):
                coord_main, coord_candidate, distance = shortest_distance(
                    main_component, merge_candidate
                )
                if distance < best_distance:
                    best_candidate_idx = candidate_idx
                    best_coord_main = coord_main
                    best_coord_candidate = coord_candidate

            tsprint("Found best candidate..")
            tsprint(
                f"Drawing line between {best_coord_main} and {best_coord_candidate}"
            )
            side_lines = side_lines = np.maximum(
                side_lines,
                draw_line(ureter_components, *best_coord_main, *best_coord_candidate),
            )

            tsprint("Adding component to main component..")
            main_component += comps[best_candidate_idx]

            tsprint("Removing best candidate from unmerged components..")
            merged_components.append(comps[best_candidate_idx])
            # del comps[best_candidate_idx]
            comps = [comps[i] for i in range(len(comps)) if i != best_candidate_idx]

        tsprint(f"All components merged for ureter on {side} side.")

        tsprint("Connecting ureter to nearest kidney component..")
        kidney_comps = kidneys_by_side[side]
        best_candidate_idx = None
        best_distance = 9999999
        best_coord_main = None
        best_coord_candidate = None
        for candidate_idx, merge_candidate in enumerate(kidney_comps):
            coord_main, coord_candidate, distance = shortest_distance(
                main_component, merge_candidate
            )
            if distance < best_distance:
                best_candidate_idx = candidate_idx
                best_coord_main = coord_main
                best_coord_candidate = coord_candidate

        if best_coord_candidate is None:
            tsprint(f"No {side} kidney detected, skipping..")
        else:
            tsprint(f"Found best candidate for {side} kidney..")
            tsprint(
                f"Drawing line between {best_coord_main} and {best_coord_candidate}"
            )
            side_lines = side_lines = np.maximum(
                side_lines,
                draw_line(ureter_components, *best_coord_main, *best_coord_candidate),
            )

        tsprint("Connecting ureter to nearest bladder component..")
        best_candidate_idx = None
        best_distance = 9999999
        best_coord_main = None
        best_coord_candidate = None
        for candidate_idx, merge_candidate in enumerate(bladder_comps):
            coord_main, coord_candidate, distance = shortest_distance(
                main_component, merge_candidate
            )
            if distance < best_distance:
                best_candidate_idx = candidate_idx
                best_coord_main = coord_main
                best_coord_candidate = coord_candidate

        if best_coord_candidate is None:
            tsprint("No bladder detected, skipping..")
        else:
            tsprint(f"Found best candidate for bladder..")
            tsprint(
                f"Drawing line between {best_coord_main} and {best_coord_candidate}"
            )
            side_lines = np.maximum(
                side_lines,
                draw_line(ureter_components, *best_coord_main, *best_coord_candidate),
            )
        side_lines = np.clip(side_lines, 0, 1)
        # im.to_sitk(side_lines, save_as="tmp/lines.nii.gz", ref_img=img_s)

        side_lines = dilate_lines(side_lines, 40, spacing)
        # im.to_sitk(side_lines, save_as="tmp/dilated_lines.nii.gz", ref_img=img_s)
        tsprint(f"Done with {side} side.")

        lines += side_lines

    tsprint("Exporting full segmentation..")
    lines = np.clip(lines, 0, 1)
    new_segmentation = lines * 4.0

    for kidney in kidneys_by_side["left"] + kidneys_by_side["right"]:
        dilated_mask = dilate_mask(kidney, 10, spacing)

        new_segmentation[dilated_mask > 0.5] = 1.0

    for bladder in bladder_comps:
        dilated_mask = dilate_mask(bladder, 10, spacing)

        new_segmentation[dilated_mask > 0.5] = 3.0

    for urether in merged_components:
        dilated_mask = dilate_mask(urether, 20, spacing)

        new_segmentation[dilated_mask > 0.5] = 2.0

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
    SEG_OUTPUT_DIR = "hema_ct/final/method3/segmentations/"
    MASK_OUTPUT_DIR = "hema_ct/final/method3/masks/"
    MASKED_SCAN_OUTPUT_DIR = "hema_ct/final/method3/masked_scans/"

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

        tsprint("Processing..")
        method_3(
            seg_path,
            scan_path,
            SEG_OUTPUT_DIR + basename,
            MASK_OUTPUT_DIR + basename,
            MASKED_SCAN_OUTPUT_DIR + basename,
        )
