import numpy as np
import nibabel as nib
from nilearn import plotting as ni_plt


def draw_slice_plot(img, affine, cuts=None, **kwargs):
    if cuts == None:
        cuts = np.flatnonzero(img.any(axis=(0,1))).astype(int)

    slice_positions = np.asarray(range(-72, 104, 5))
    map_nifti = nib.nifti1.Nifti1Image(img, affine)
    return ni_plt.plot_stat_map(map_nifti, display_mode="z", cut_coords=slice_positions[cuts], **kwargs), map_nifti
