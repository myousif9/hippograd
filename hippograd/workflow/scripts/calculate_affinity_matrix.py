import nibabel as nib
import numpy as np
import math
import logging

from brainspace.utils.parcellation import reduce_by_labels
from scipy.spatial.distance import pdist, squareform

def pull_data(cifti,structure):
  """Loads left and right hemispheres of specified structure data from cifti file

  Args:
      cifti (string): File path to cifti file.
      structure (string): Structure to pull data for.

  Returns:
      np.array: Left and right (in that order) cifti data for specified structure.
  """

  cifti_file = nib.load(cifti)
  header_data = cifti_file.header.get_axis(1)
  for brain_structure in header_data.iter_structures():
      if brain_structure[0] == f'CIFTI_STRUCTURE_{structure.upper()}_LEFT':
          start = brain_structure[1].start
      if brain_structure[0] == f'CIFTI_STRUCTURE_{structure.upper()}_RIGHT':
          stop = brain_structure[1].stop
  data = cifti_file.get_fdata()[:,start:stop]
  return np.transpose(data)

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
  logfile = snakemake.log[0]


  compute_affinity_matrix(rfmri_hipp_file, rfmri_ctx_file, affinity_matrix_output, correlation_matrix_output,logfile=logfile)
