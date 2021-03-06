import nibabel as nib
import numpy as np
import logging

from brainspace.utils.parcellation import reduce_by_labels
from utilities import loadciftiLRstruct

def generate_correlation_map(x, y):
  """Correlate each n with each m.

  Args:
      x (np.array): Shape N X T.
      y (np.array): Shape M X T.

  Returns:
      np.array: N X M array in which each element is a correlation coefficient.
  """
  
  mu_x = x.mean(1)
  mu_y = y.mean(1)
  n = x.shape[1]
  if n != y.shape[1]:
      raise ValueError('x and y must ' +
                        'have the same number of timepoints.')
  s_x = x.std(1, ddof=n - 1)
  s_y = y.std(1, ddof=n - 1)
  cov = np.dot(x,
                y.T) - n * np.dot(mu_x[:, np.newaxis],
                                mu_y[np.newaxis, :])
  np.seterr(divide='ignore',invalid='ignore')
  return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def compute_correlation_matrix(rfmri_hippo_path, rfmri_ctx_path, hippo_hemi=None, lateralization='ipsi', atlas_path=None,logfile=None):
  """_summary_

  Args:
      rfmri_hippo_path (string): Define path to hippocampus gifti file.
      rfmri_ctx_path (string): Define path to cortex cifti file. 
      atlas_path (string, optional): Define path to cortex parcellation. Defaults to None.
      logfile (string, optional): Define path to output logfile. Defaults to None.

  Returns:
      numpy array: Correlation matrix 
  """

  logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 
  rfmri_hipp_gii  = nib.load(rfmri_hippo_path)
  rfmri_hipp_data_rest = np.zeros((len(rfmri_hipp_gii.darrays[0].data),len(rfmri_hipp_gii.darrays)))

  for i in range(0,len(rfmri_hipp_gii.darrays)):
    rfmri_hipp_data_rest[:,i] = rfmri_hipp_gii.darrays[i].data

  rfmri_ctx_data_rest = loadciftiLRstruct(rfmri_ctx_path,'cortex')
  logging.info('Cortex data loaded.')

  ctxLR = np.split(rfmri_ctx_data_rest,2,axis=1)
  if atlas_path != None:
    atlas = np.load(atlas_path)
    if 'ipis' in lateralization:
      ctx = ctxLR[0] if hippo_hemi == 'L' else ctxLR[1]
      atlasLR = np.split(atlas,2)
      atlas = atlasLR[0] if hippo_hemi == 'L' else atlasLR[1]
    elif 'contra' in lateralization:
      ctx = ctxLR[0] if hippo_hemi == 'R' else ctxLR[1]
      atlasLR = np.split(atlas,2)
      atlas = atlasLR[0] if hippo_hemi == 'R' else atlasLR[1]
    else:
      ctx = rfmri_ctx_data_rest
    
    ctx = reduce_by_labels(ctx,atlas).T
    logging.info('Cortex data reduced to atlas parcellation.')

  else:
    if 'ipis' in lateralization:
      ctx = ctxLR[0] if hippo_hemi == 'L' else ctxLR[1]

    elif 'contra' in lateralization:
      ctx = ctxLR[0] if hippo_hemi == 'R' else ctxLR[1]

    else:
      ctx = rfmri_ctx_data_rest
    
    ctx = ctx.T

  # Compute hipp vertex-wise correlation matrix first
  correlation_matrix = generate_correlation_map(rfmri_hipp_data_rest,ctx)
  correlation_matrix = np.nan_to_num(correlation_matrix)
  logging.info('Correlation matrix computed.')

  return correlation_matrix

if __name__ == '__main__':

  # define variable for input rfMRI data
  fmri_hipp_path = snakemake.input.rfmri_hipp
  fmri_ctx_path  = snakemake.input.rfmri_ctx
  
  # define variables for output file paths
  correlation_matrix_output = snakemake.output.correlation_matrix
  logfile = snakemake.log[0]
  
  logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 

  # defining variables for correlation calculation parameters
  atlas_path = snakemake.params.parcellation
  ctx_lateralization = snakemake.params.cortex_lateralization
  logging.info('I/O variables defined.')  

  corr_mat = compute_correlation_matrix(fmri_hipp_path,fmri_ctx_path,atlas_path=atlas_path,lateralization=ctx_lateralization,logfile=logfile)
  
  np.save(correlation_matrix_output,corr_mat)
  logging.info('Correlation matrix computed and saved.')