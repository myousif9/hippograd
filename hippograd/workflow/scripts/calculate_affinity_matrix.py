import nibabel as nib
import numpy as np
import math
import logging

from brainspace.utils.parcellation import reduce_by_labels
from scipy.spatial.distance import pdist, squareform
from scripts.utilities import pull_data, generate_correlation_map

def compute_affinity_matrix(rfmri_hipp_file, rfmri_ctx_file, affinity_matrix_output,correlation_matrix_output,atlas_file=None,logfile=None):
  """Compute affinity matrix

  Args:
      rfmri_hipp_file (gifti): Path to fMRI data projected to hippocampal surface.
      rfmri_ctx_file (cifti): Path to fMRI cortex data.
      affinity_matrix_output (string): Affinity matrix output path.
      correlation_matrix_output (string): Correlation matrix output path.
  """
  logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 

  rfmri_hipp_gii  = nib.load(rfmri_hipp_file)
  rfmri_hipp_data_rest = np.zeros((len(rfmri_hipp_gii.darrays[0].data),len(rfmri_hipp_gii.darrays)))

  for i in range(0,len(rfmri_hipp_gii.darrays)):
      rfmri_hipp_data_rest[:,i] = rfmri_hipp_gii.darrays[i].data
  
  rfmri_ctx_data_rest = pull_data(rfmri_ctx_file,'cortex')
  logging.info('Cortex data loaded.')

  if atlas_file != None:
    atlas = np.load(atlas_file)
    rfmri_ctx_data_rest = reduce_by_labels(rfmri_ctx_data_rest,atlas)
    logging.info('Cortex data reduced to atlas parcellation.')
    
  # Compute hipp vertex-wise correlation matrix first
  correlation_matrix = generate_correlation_map(rfmri_hipp_data_rest,rfmri_ctx_data_rest)
  correlation_matrix = np.nan_to_num(correlation_matrix)
  logging.info('Correlation matrix computed.')

  # Save to npy file
  np.save(correlation_matrix_output, correlation_matrix)
  logging.info('Correlation matrix saved.')

  # Transform correlation matrix to cosine similarity and then normalized angle matrix
  dist_upper        = pdist(correlation_matrix,'cosine')
  cosine_similarity = 1-squareform(dist_upper)
  cosine_similarity = np.nan_to_num(cosine_similarity)
  norm_angle_matrix = 1-(np.arccos(cosine_similarity)/math.pi)
  logging.info('Affinity matrix computed.')

  # Save to npy file
  np.save(affinity_matrix_output, norm_angle_matrix)
  logging.info('Affinity matrix saved.')

if __name__ == '__main__':

  # Load hippocampal rfMRI data
  rfmri_hipp_file = snakemake.input.rfmri_hipp
  rfmri_ctx_file  = snakemake.input.rfmri_ctx
  affinity_matrix_output = snakemake.output.affinity_matrix
  correlation_matrix_output = snakemake.output.correlation_matrix
  atlas_path = snakemake.params.parcellation
  logfile = snakemake.log[0]


  compute_affinity_matrix(rfmri_hipp_file, rfmri_ctx_file, affinity_matrix_output, correlation_matrix_output, atlas=atlas_path, logfile=logfile)
