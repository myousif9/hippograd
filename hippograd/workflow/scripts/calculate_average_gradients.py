from operator import index
import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps
import os
import logging

logfile=snakemake.log[0]
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 

# Get number of subjects
affinity_mat_list = snakemake.input.affinity_matrix
n_subjects = len(affinity_mat_list)
# Load subject info
# subject_info = pd.read_table(snakemake.input.subject_info, header=0)

# Load first subject's affinity matrix for shape
affinity_matrix_data = np.load(affinity_mat_list[0])
logging.info('First subject affinity matrix loaded.')

# Create empty matrix to fill
affinity_matrix_concat = np.zeros((
    affinity_matrix_data.shape[0],
    affinity_matrix_data.shape[1],
    n_subjects)
)

index_delete = []

# Fill empty matrix using subjects' affinity matrices
for s, affinity_path in enumerate(affinity_mat_list):
    if os.path.exists(snakemake.input.affinity_matrix[s]):
        affinity_matrix_concat[:,:,s] = np.load(snakemake.input.affinity_matrix[s])
        logging.info('%s file exists and included in average.',affinity_path)

    else:
        index_delete.append(s)
        logging.info('%s file does not exist and is not included in average.',affinity_path)

if len(index_delete)>0:
    affinity_matrix_concat = np.delete(affinity_matrix_concat,index_delete,2)

# Average affinity matrices
affinity_matrix_avg = np.mean(affinity_matrix_concat,2)
logging.info('Average affinity matrix computed.')

# Save matrix
np.save(snakemake.output.affinity_matrix, affinity_matrix_avg)
logging.info('Affinity matrix saved.')

# Calculate gradients based on average normalized angle matrix
# Kernel = none as input matrix is already affinity matrix
gm = GradientMaps(n_components=snakemake.params.n_gradients, kernel=None, random_state=0)
gm.fit(affinity_matrix_avg, diffusion_time=0)
logging.info('Average gradients computed.')

# Save gradients to gifti file
gii = nib.gifti.GiftiImage()

for g in range(0,len(gm.gradients_.T)):
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=gm.gradients_.T[g].astype(np.float32),
            meta={
                'AnatomicalStructurePrimary':'Cortex_Left' if snakemake.wildcards.hemi == 'L' else 'Cortex_Right',
                'Name':'Gradient {}'.format(g+1)
                }
            )
    ) 

nib.save(gii, snakemake.output.gradient_maps)
logging.info('Average gradient file saved.')