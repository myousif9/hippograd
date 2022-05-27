import nibabel as nib
import numpy as np
import logging

logfile = snakemake.log[0]
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 

gii = nib.load(snakemake.input.surf)
varr = gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
V = varr.data
farr = gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0]
F = farr.data
logging.info('Surface data loaded.')

if np.isnan(np.sum(V)):
    logging.info('NaNs found!')
else:
    logging.info('No NaNs found!')

# most nans should be just isolated points, but in case there is an island of nans this will erode it, replacing with decent (but not perfect) guesses of where vertices should be
while np.isnan(np.sum(V)):
    # index of vertices containing nan
    i = np.where(np.isnan(V))
    ii = np.unique(i[0])
    # replace with the nanmean of neighbouring vertices
    newV = V
    for n in ii:
        f = np.where(F==n)
        v = F[f[0]]
        vv = np.unique(v)
        newV[n,:] = np.nanmean(V[vv,:],0)
    V = newV

nib.save(gii,snakemake.output.surf)
logging.info('NaNless surface saved!')