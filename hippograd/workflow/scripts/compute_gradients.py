import numpy as np
import logging
from brainspace.gradient import GradientMaps
from utilities import hippograd2gifti, density_interp

# define variable for input rfMRI data
corr_mat_path = snakemake.input.correlation_matrix
# define variables for output file paths
gradient_output = snakemake.output.gradient_maps
lambdas = snakemake.output.lambdas
logfile = snakemake.log[0]

logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S: %p') 

# defining variables for gradient calculation parameters
hemi = snakemake.wildcards.hemi
reference_grad = snakemake.params.refgradL if hemi == 'L' else snakemake.params.refgradR
n_gradients = snakemake.params.n_gradients
kernel = snakemake.params.kernel
embedding = snakemake.params.embedding
resource_dir = snakemake.params.resource_dir

align = snakemake.params.align
align = None if align == 'none' else align

density = snakemake.wildcards.density
ctx_lateralization = snakemake.params.cortex_lateralization
logging.info('I/O variables defined.')  

corr_mat = np.load(corr_mat_path)
logging.info('Correlation matrix loaded')

reference_grad_data = np.load(reference_grad)

if density != '0p5mm':
  reference_grad_data = density_interp('0p5mm',density, reference_grad_data,resources_dir=resource_dir)[0]
  logging.info('Reference gradient resampled from density 0p5mm and %s',density)

gm = GradientMaps(n_components=n_gradients, approach=embedding, kernel=kernel, alignment=align)
gm.fit(corr_mat, reference = reference_grad_data, sparsity=0),
logging.info('Gradient maps computed.')

np.save(lambdas, gm.lambdas_)
logging.info('Lambdas saved.')

if align == 'procrustes':
  gii = hippograd2gifti(gm.aligned_,hemi)
else:
  gii = hippograd2gifti(gm.gradients_,hemi)

gii.to_filename(gradient_output)
logging.info('Gradient maps saved to %s.', gradient_output)