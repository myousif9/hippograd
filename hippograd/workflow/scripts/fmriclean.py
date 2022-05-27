from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.image import clean_img, smooth_img
from nilearn.signal import clean
import nibabel as nib
import pandas as pd
import logging



def clean_fmri(img_path, mask_path=None, clean_strategy='simple', logfile=None, **kwargs):
    """Clean fmri image based on predefined strategy, 
    
    Args:
        img_path (string): Path to fmri image in fmriprep directory
        mask_path (string): Path to fmri mask in fmriprep directory default is None but mask is recommended
        clean_strategy (string): Input for predefined cleaning strategy, options: 'simple','srubbing','compcor','ica_aroma'
        kwargs: For more customized control over cleaning, you can pass named parameters to nilearn function clean_img (used for volume) or clean (used for surface)

    Returns:
        Niimg-like object: cleaned fmri image, same shape as inputed image.
    """
    logging.basicConfig(filename=logfile,level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p')

    confounds, sample_mask  = load_confounds_strategy(img_path, denoise_strategy = clean_strategy)
    logging.info('Confounds loaded!')
    
    if confounds.isnull().values.any():
        logging.critical('There are NaNs in your confounds! Will likely cause downstream errors!')

    if '.nii.gz' in img_path:
        img = smooth_img(img_path,None)
        mask = smooth_img(mask_path,None) if mask_path != None else None
        logging.info('Volume fMRI image and mask loaded!')

        fmri_clean = clean_img(img, confounds = confounds, mask_img=mask, **kwargs)
    else:
        img = nib.load(img_path)
        img_data = img.get_fdata()
        logging.info('Surface fMRI loaded!')

        fmri_clean_data = clean(img_data,confounds=confounds,**kwargs)
        fmri_clean = nib.cifti2.Cifti2Image(fmri_clean_data,header=img.header,nifti_header=img.nifti_header)
        fmri_clean.update_headers()
    
    logging.info('fMRI cleaned!')

    return fmri_clean

if __name__ == '__main__':
    logfile = snakemake.log[0]

    vol_clean = clean_fmri(snakemake.input.bold_vol, mask_path=snakemake.input.mask, clean_strategy=snakemake.params.strategy,logfile=logfile)
    vol_clean.to_filename(snakemake.output.fmri_volume)

    surf_clean = clean_fmri(snakemake.input.bold_surf, clean_strategy=snakemake.params.strategy,logfile=logfile)
    surf_clean.to_filename(snakemake.output.fmri_surf)