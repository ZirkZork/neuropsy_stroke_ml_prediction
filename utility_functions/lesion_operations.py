#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import math
import os
import re
import nibabel as nib

def check_overlap(map_a:np.ndarray, map_b:np.ndarray):
    binary_map_a = abs(np.sign(map_a.astype(float)))
    binary_map_b = abs(np.sign(map_b.astype(float)))
    overlap_map = binary_map_a * binary_map_b

    map_a_size = np.sum(binary_map_a)
    map_b_size = np.sum(binary_map_b)

    overlap_size = np.sum(overlap_map)

    return overlap_size, overlap_size/map_a_size, overlap_size/map_b_size

def load_lesion_maps(subject_ids, data_folder:str, slice_spacing:int = 5, slice_offset_from_center:int = -2):
    
    ## load files

    n_subjects = len(subject_ids)

    found_lesion_maps = []
    found_subject_ids = []

    map_affine = None
    map_dimensions = None

    for id in subject_ids:
        lesion_map, affine = _read_voi_file(id, data_folder)

        if lesion_map is None:
            print(f"No lesion map found for {id}.")
            continue

        found_lesion_maps.append(lesion_map)
        found_subject_ids.append(id)

        if map_affine is None:
            map_affine = affine
        if not np.all(affine == map_affine):
            print(f"\"{id}\" - non matching affines - got: {affine} and expected: {map_affine}")

        if map_dimensions is None:
            map_dimensions = lesion_map.shape
        if not np.all(lesion_map.shape == map_dimensions):
            print(f"\"{id}\" - non matching dimensions - got: {lesion_map.shape} and expected: {map_dimensions}")
    
    n_subjects_found = len(found_lesion_maps)

    print (f"{n_subjects_found} out of {n_subjects:d} lesion map{"s" if n_subjects > 1 else ""} loaded.")

    lesion_maps = np.asarray(found_lesion_maps)

    ## extract lesion information
    n_slices = math.ceil(map_dimensions[2] / slice_spacing)
    slice_dimensions = map_dimensions[:2] + (n_slices,)

    slice_lesion_maps = np.zeros((n_subjects_found,) + slice_dimensions)

    affine_z_offset = map_affine[2,3]

    slice_affine = np.copy(map_affine)
    slice_affine[2,2] = slice_spacing
    slice_affine[2,3] = affine_z_offset - (affine_z_offset - slice_offset_from_center)%slice_spacing

    center_slice = apply_inverse_affine(np.asarray([[0, 0, slice_offset_from_center]]), map_affine)[0, 2]

    lesion_layers_sub, lesion_layers = np.nonzero(lesion_maps.any(axis=(1,2)))

    for sub, layer in zip(lesion_layers_sub, lesion_layers):
        on_correct_slice = (layer - center_slice) % slice_spacing == 0
        mni_z = apply_affine(np.asarray([[0, 0, layer]]), map_affine)[0, 2]

        if not on_correct_slice:
            n_voxels = lesion_maps[sub, :, :, layer].sum()
            print(f"\"{subject_ids[sub]}\"({sub}) - {n_voxels} voxels mapped on incorrect layer {layer} (mni z: {mni_z})")
            continue

        layer_slice_space = int(apply_inverse_affine(np.asarray([[0, 0, mni_z]]), slice_affine)[0, 2])
        slice_lesion_maps[sub, :, :, layer_slice_space] = lesion_maps[sub, :, :, layer]


    return slice_lesion_maps, slice_dimensions, slice_affine, found_subject_ids

def _read_voi_file (subject_id:str, data_folder:str):
    # dataset specific subject labels: first 2 letters of last name + first 2 letters of first name + day and month of birth (4 digits) + "w" or "m" for gender
    voi_file_re = re.compile(r".*(?P<subject_id>\w{4}\d{4}(m|w)).*(?P<file_extension>\.nii(\.gz)?|\.voi)")
    file_subject_id = None
    file_extension = None

    for path, subdirs, files in os.walk(data_folder):
        for file_name in files:
            file_re_match = voi_file_re.fullmatch(file_name)
            if file_re_match is None:
                continue

            file_subject_id = file_re_match.group("subject_id")
            file_extension = file_re_match.group("file_extension")

            if file_subject_id.capitalize() != subject_id.capitalize():
                continue

            if file_extension == ".voi":

                old_file_name = file_name
                file_name = re.sub(r"\.voi", ".nii.gz", file_name)
                os.rename(os.path.join(path, old_file_name), os.path.join(path, file_name))
            
            return load_nifti (os.path.join(path, file_name))
                
    return None, None


def load_nifti (file_apth):
    nifti = nib.load(file_apth)

    affine = nifti.affine
    voxel_matrix = nifti.get_fdata()
    
    return voxel_matrix, affine

def save_as_nifti (voxel_matrix, affine, filepath):
    nifti = nib.nifti1.Nifti1Image(voxel_matrix, affine)
    
    nib.save(nifti, filepath)

def interpolate_lesion_map (lesion_map:np.ndarray, slice_gap:int = 5, slice_axis:int = 2, cutoff:float = 0.45):

    kernel_size = slice_gap * 2 - 1
    interpolation_kernel = np.zeros((kernel_size, kernel_size, kernel_size))

    for i in range(kernel_size):
        outer_slice_tuple = [slice(None), slice(None), slice(None)]
        outer_slice_tuple[slice_axis] = i
        outer_slice_tuple = tuple(outer_slice_tuple)

        dist_from_center = abs(slice_gap - i - 1)
        for (j, k), x in np.ndenumerate(interpolation_kernel[outer_slice_tuple]):
            inner_slice_tuple = [slice(None), slice(None), slice(None)]
            indices=[ind if ind != slice_axis else None for ind in range(2)]
            for l, axis in zip(indices, [j, k]):
                inner_slice_tuple[l] = axis
            inner_slice_tuple[slice_axis] = i
            inner_slice_tuple = tuple(inner_slice_tuple)

            point_dist_from_center = np.linalg.norm(np.asarray([slice_gap - j - 1, slice_gap - k - 1]))

            interpolation_kernel[j, k, i] = float(point_dist_from_center <= dist_from_center)

        interpolation_kernel[:, :, i] /= interpolation_kernel[:, :, i].sum()

        if dist_from_center != 0:
            interpolation_kernel[:, :, i] /= 2
        
    kernel_size = slice_gap * 2 - 1
    liear_interpolation_kernel = np.zeros((kernel_size, kernel_size, kernel_size))

    center = kernel_size // 2

    slice_tuple = [center, center, center]
    slice_tuple[slice_axis] = slice((kernel_size - slice_gap) // 2, (kernel_size + slice_gap) // 2)
    slice_tuple = tuple(slice_tuple)
    liear_interpolation_kernel[slice_tuple] = 1
    
    gaussian_c = 1

    gaussian_kernel = np.zeros((kernel_size, kernel_size, kernel_size))
    for index in np.ndindex(gaussian_kernel.shape):
        ind_array = np.asarray(index)
        ind_array -= center
        dist = np.linalg.norm(ind_array)
        value = math.exp(- dist ** 2 / (2 * gaussian_c ** 2))
        gaussian_kernel[index] = value

    lesion_map_interpolated = scipy.signal.convolve(lesion_map, interpolation_kernel, mode="same")
    lesion_map_interpolated_threshholded = (lesion_map_interpolated >= cutoff).astype(float)

    return lesion_map_interpolated_threshholded


def apply_affine (coordinates:np.ndarray, affine:np.ndarray):

    n_coordinates = coordinates.shape[0]
    n_dimensions = coordinates.shape[1]

    assert(affine.shape[0] == n_dimensions + 1 and affine.shape[1] == n_dimensions + 1)

    coordinates_expanded = np.append(coordinates, np.ones((n_coordinates, 1)), axis=1)
    coordinates_transformed = (affine.astype(float) @ coordinates_expanded.T).T

    return coordinates_transformed[:,:3]

def apply_inverse_affine (coordinates:np.ndarray, affine:np.ndarray):

    h = affine.shape[0] - 1
    w = affine.shape[1] - 1

    M = affine[:h, :w]
    b = affine[:h, w]

    invM = np.linalg.inv(M)
    invb = -invM.dot(b)

    inverse_affine = np.copy(affine.astype(float))
    inverse_affine[:h,:w] = invM
    inverse_affine[:h, w] = invb

    return apply_affine(coordinates, inverse_affine)