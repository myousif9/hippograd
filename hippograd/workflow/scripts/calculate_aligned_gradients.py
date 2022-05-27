import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps


logfile=snakemake.log[0]
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 

# Load affinity matrix
affinity_matrix = np.load(snakemake.input.affinity_matrix)
logging.info('Subject affinity matrix loaded.')

# Load average affinity matrix
avg_affinity_matrix = np.load(snakemake.input.avg_affinity_matrix)
logging.info('Average affinity matrix loaded.')

# Calculate gradients based on normalized angle matrix
# Kernel = none as input matrix is already affinity matrix
gp = GradientMaps(n_components=snakemake.params.n_gradients, kernel=None, alignment='procrustes', random_state=0)
gp.fit([avg_affinity_matrix, affinity_matrix], diffusion_time=0)
logging.info('Subject gradients computed.')

# Save aligned gradients to gifti file
gii = nib.gifti.GiftiImage()
gii_unaligned = nib.gifti.GiftiImage()

for g in range(0,snakemake.params.n_gradients):
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=gp.aligned_[1][:,g].astype(np.float32),
            meta={
                'AnatomicalStructurePrimary':'Cortex_Left' if snakemake.wildcards.hemi == 'L' else 'Cortex_Right',
                'Name':'Gradient {}'.format(g+1)
                }
            )
    ) 

    gii_unaligned.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=gp.gradients_[1][:,g].astype(np.float32),
            meta={
                'AnatomicalStructurePrimary':'Cortex_Left' if snakemake.wildcards.hemi == 'L' else 'Cortex_Right',
                'Name':'Gradient {}'.format(g+1)
                }
            )
    ) 


nib.save(gii, snakemake.output.gradient_maps)
nib.save(gii_unaligned, snakemake.output.gradient_unaligned)
logging.info('Subject aligned and unaligned gradients saved.')