import nibabel as nib
import numpy as np
import pandas as pd
import os
import copy
from scipy.interpolate import griddata
from yaml import load



def gifti2csv(gii_file, out_file, itk_lps = True):
        gii = nib.load(gii_file)
        data = gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data

        if itk_lps:  # ITK: flip X and Y around 0
            data[:, :2] *= -1

        # antsApplyTransformsToPoints requires 5 cols with headers
        csvdata = np.hstack((data, np.zeros((data.shape[0], 3))))
        
        np.savetxt(
            out_file,
            csvdata,
            delimiter=",",
            header="x,y,z,t,label,comment",
            fmt=["%.5f"] * 4 + ["%d"] * 2,
        )

def csv2gifti(csv_file, gii_file, out_file, itk_lps = True):
    gii = nib.load(gii_file)
    vertices = gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data 
    
    data = np.loadtxt(
        csv_file, delimiter=",", skiprows=1, usecols=(0, 1, 2)
    )

    if itk_lps:  # ITK: flip X and Y around 0
        data[:, :2] *= -1
    
    new_gii = nib.gifti.GiftiImage(header=gii.header,meta = gii.meta)

    new_gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=data[:, :3].astype(vertices.dtype),
            intent='NIFTI_INTENT_POINTSET'
        )
    )
    
    new_gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
                data = faces.astype(faces.dtype),
                intent = 'NIFTI_INTENT_TRIANGLE'
            )
    )
    
    new_gii.to_filename(out_file)

def fmri_path_cohort(cohort_path):
    df = pd.read_csv(cohort_path)

    path = []
    for idx, item in enumerate([ x for x in df.columns.to_list() if 'id' in x]):
        item_list = df[item].to_list()
        run = 0
        ses = 0
        if 'run' in item_list[0]:
            run = item_list[np.argmax([int(item.replace('run-', '')) for item in item_list])]
            path.append(run)
        else:
            path.append(item_list[idx])

    root = '/'.join(path)
    file_prefix = '_'.join(path)  
    file_vol =  file_prefix + '_residualised.nii.gz'
    file_surf =  file_prefix + '_residualised_space-fsLR_den-91k_bold.dtseries.nii'
    return os.path.join(root, 'regress', file_vol), os.path.join(root, 'regress', file_surf)

def hippograd2gifti(gradients, hemi):
    """Assiging gradients from numpy output to easily saveable gifti format

    Args:
        gradients (numpy array): _description_
        hemi (string): Specifies hemisphere for generating 

    Returns:
        gifti: _description_
    """
    hemi = hemi.strip().upper()
    gradients = gradients.T

    gii = nib.gifti.GiftiImage()

    for g in range(0,len(gradients)):
        gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data = gradients[g].astype(np.float32),
                meta = {
                    'AnatomicalStructurePrimary':'CORTEX_LEFT' if hemi == 'L' else 'CORTEX_RIGHT',
                    'Name':'Gradient {}'.format(g+1)
                    }
                )
        )
    return gii

# from neurohackacademy tutorial
def loadcifti_surf(data, surf_name):
    """Loading structure specific surface data from cifti files into numpy arrays

    Args:
        data (Cifti2): Cifti2 image (output of loading .dtseries.nii)
        surf_name (string): Input cifti structure to extract ie. "CORTEX_LEFT", "CORTEX_RIGHT", etc

    Raises:
        ValueError: String

    Returns:
        numpy array: contains  
    """
    axis = data.header.get_axis(1)
    data = data.get_fdata()
    surf_name = 'CIFTI_STRUCTURE_'+surf_name.strip().upper()
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")

def loadciftiLRstruct(cifti_path,struct):
    """Load left and right hemispheres of structure

    Args:
        cifti_path (string): Path to cifti file.
        struct (string): Specify structure to load.

    Returns:
        numpy array: _description_
    """
    cifti_data = nib.load(cifti_path)
    struct = struct.strip().upper()
    structL = loadcifti_surf(cifti_data, struct + '_LEFT')
    structR = loadcifti_surf(cifti_data, struct + '_RIGHT')
    return np.concatenate([structL,structR]).T

def fetch_atlas_path(atlas,n_parc,parc_dir):
    """Fetching path the atlas. All parcellation where downloaded via brainstat package (https://github.com/MICA-MNI/BrainStat).

    Args:
        atlas (string): Parcellation selection eg. schaefer cammoun, glasser, yeo
        n_parc (integer): Number of parcels.
        parc_dir (string): Path to directory where parcellations are stored.

    Raises:
        ValueError: Invalid atlas.
        ValueError: Invalid number of parcels.
        ValueError: Path to parcellation directory does not exist.

    Returns:
        string: Path to selected parcelation.
    """

    atlas_dict = {'schaefer':[100, 200, 300, 400, 500, 600, 800, 1000],'cammoun':[33, 60, 125, 250, 500],'glasser':[360],'yeo':[7,17]}
    
    if atlas not in atlas_dict.keys():
        raise ValueError( f'{atlas} is not a valid atlas, valid atlases are:'+', '.join(atlas_dict)+'.')
    elif n_parc not in atlas_dict[atlas]:
        n_parc_str = str(n_parc)
        raise ValueError(f'{n_parc_str} is not a valid number of parcels for the {atlas} atlas.')
    elif os.path.exists(parc_dir) == False:
        raise ValueError(f'{parc_dir} path does not exist.')
    else:
        return os.path.join(parc_dir,atlas+str(n_parc)+'.npy')

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

# borrowed from Jordan Dekraker hippunfold_toolbox url: https://github.com/jordandekraker/hippunfold_toolbox/blob/caa60bec5ec28073acd3f70b299e882826911f97/Python/utils.py
def fillnanvertices(F,V):
    '''Fills NaNs by iteratively nanmean nearest neighbours until no NaNs remain. Can be used to fill missing vertices OR missing vertex cdata.'''
    Vnew = copy.deepcopy(V)
    while np.isnan(np.sum(Vnew)):
        # index of vertices containing nan
        vrows = np.unique(np.where(np.isnan(Vnew))[0])
        # replace with the nanmean of neighbouring vertices
        for n in vrows:
            frows = np.where(F == n)[0]
            neighbours = np.unique(F[frows,:])
            Vnew[n] = np.nanmean(Vnew[neighbours], 0)
    return Vnew

def density_interp(indensity, outdensity, cdata, method='nearest', resources_dir='../resources'):
    '''interpolates data from one surface density onto another via unfolded space
    Inputs:
      indensity: one of '0p5mm', '1mm', '2mm', or 'unfoldiso
      outdensity: one of '0p5mm', '1mm', '2mm', or 'unfoldiso
      cdata: data to be interpolated (same number of vertices, N, as indensity)
      method: 'nearest', 'linear', or 'cubic'. 
      resources_dir: path to hippunfold resources folder
    Outputs: 
      interp: interpolated data
      faces: face connectivity from new surface density'''
    
    VALID_STATUS = {'0p5mm', '1mm', '2mm', 'unfoldiso'}
    if indensity not in VALID_STATUS:
        raise ValueError("results: indensity must be one of %r." % VALID_STATUS)
    if outdensity not in VALID_STATUS:
        raise ValueError("results: outdensity must be one of %r." % VALID_STATUS)
    
    # load unfolded surfaces for topological matching
    startsurf = nib.load(f'{resources_dir}/unfold_template_hipp/tpl-avg_space-unfold_den-{indensity}_midthickness.surf.gii')
    vertices_start = startsurf.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    targetsurf = nib.load(f'{resources_dir}/unfold_template_hipp/tpl-avg_space-unfold_den-{outdensity}_midthickness.surf.gii')
    vertices_target = targetsurf.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = targetsurf.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data

    # interpolate
    interp = griddata(vertices_start[:,:2], values=cdata, xi=vertices_target[:,:2], method=method)
    # fill any NaNs
    interp = fillnanvertices(faces,interp)
    return interp,faces,vertices_target

def smask_cifti(img,smask):
    """Function for applying sample mask to cifti fMRI data.

    Args:
        img (Cifti Niimg-like object): Cifti fMRI data.
        smask (numpy array): Array of indicies of fMRI time series to sample.

    Returns:
        Cifti Nii-like object: Sample masked fMRI data. 
    """
    if isinstance(smask,type(None)):
        return img
    else:
        hdr = img.header
        series = hdr.get_axis(0)
        series.size = np.size(smask)
        series_mapping = series.to_mapping(0)
        
        brain_model = img.header.get_axis(1)
        brain_model_mapping = brain_model.to_mapping(1)
        
        nifti_hdr = img.nifti_header
        
        mtx = nib.cifti2.Cifti2Matrix()
        mtx.append(series_mapping)
        mtx.append(brain_model_mapping)
        
        smask_hdr = nib.Cifti2Header(mtx)
        smask_data = img.get_fdata()[smask,:]

        smask_img = nib.Cifti2Image(smask_data,smask_hdr,nifti_hdr)
        smask_img.update_headers()
        
        return smask_img
    