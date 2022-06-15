from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.image import clean_img, smooth_img
from nilearn.signal import clean
import nibabel as nib
import numpy as np
import logging
from utilities import smask_cifti



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

        if isinstance(sample_mask,np.ndarray):
            smask_data = img.get_fdata()[:,:,:,sample_mask]
            img = nib.Nifti1Image(smask_data,img.affine,img.header)
            img.update_header()
            confounds = confounds.iloc[sample_mask,:]
            logging.info('Volume image sample masked.')

        fmri_clean = clean_img(img, confounds = confounds, mask_img=mask,**kwargs)
    else:
        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_hdr = img.header
        img_nii_hdr = img.nifti_header
        logging.info('Surface fMRI loaded!')

        if isinstance(sample_mask,np.ndarray):
            img_smask = smask_cifti(img,sample_mask)
            img_data = img_smask.get_fdata()
            img_hdr = img_smask.header
            img_nii_hdr = img_smask.nifti_header
            confounds = confounds.iloc[sample_mask,:]
            logging.info('Cifti fMRI sample masked!')

        fmri_clean_data = clean(img_data, confounds=confounds, **kwargs)
        fmri_clean = nib.Cifti2Image(fmri_clean_data, img_hdr, img_nii_hdr)
        fmri_clean.update_headers()
    
    logging.info('fMRI cleaned!')

    return fmri_clean

if __name__ == '__main__':
    logfile = snakemake.log[0]

    vol_clean = clean_fmri(snakemake.input.bold_vol, mask_path=snakemake.input.mask, clean_strategy=snakemake.params.strategy,logfile=logfile)
    vol_clean.to_filename(snakemake.output.fmri_volume)

    surf_clean = clean_fmri(snakemake.input.bold_surf, clean_strategy=snakemake.params.strategy,logfile=logfile)
    surf_clean.to_filename(snakemake.output.fmri_surf)