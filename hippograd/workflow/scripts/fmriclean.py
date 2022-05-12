from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.image import clean_img
import nibabel as nib
import numpy as np

def clean_fmri(img_path, clean_strategy):
    """Clean fmri image based on predefined strategy
    
    Args:
        img_path (string): Path to fmri image in fmriprep directory
        clean_strategy (string): Input for predefined 

    Returns:
        Niimg-like object: cleaned fmri image, same shape as inputed image.
    """
    confounds, sample_mask  = load_confounds_strategy( img_path, denoise_strategy = clean_strategy)
    fmri_clean = clean_img(img_path, confounds = confounds, mask_img = sample_mask)
    return fmri_clean

def save_nii(img_data, outfile_name):
    """save niimg-like data to cifti file

    Args:
        img_data (niimg-like array): fmri data to be saved
        outfile_name (string): output file name 
    """
    nii_img = nib.cifti2.Cifti2Image(dataobj = img_data)
    nii_img.update_headers()
    nib.save(nii_img, outfile_name)

def save_vol(img_data, outfile_name):
    """save niimg-like data to nifti file

    Args:
        img_data (niimg-like array): fmri data to be saved
        outfile_name (string): output file name 
    """
    vol_img = nib.Nifti1Image(img_data,np.ones(4))
    vol_img.header.get_xyztunits()
    vol_img.to_filename(outfile_name)

if __name__ == '__main__':
    
    surf_clean = clean_fmri(snakemake.input.bold_surf, snakemake.params.strategy)
    vol_clean = clean_fmri(snakemake.input.bold_vol, snakemake.params.strategy)
    
    # sav
    save_nii(surf_clean, snakemake.output.fmriclean_surf)
    save_vol(vol_clean, snakemake.output.fmriclean_vol)