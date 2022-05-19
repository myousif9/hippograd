import nibabel as nib
import numpy as np
import pandas as pd
import os

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
        gradients (_type_): _description_
        hemi (string): Specifies hemisphere

    Returns:
        gifti: _description_
    """
    gii = nib.gifti.GiftiImage()

    for g in range(0,len(gradients)):
        gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data = gradients[g].astype(np.float32),
                meta = {
                    'AnatomicalStructurePrimary':'CortexLeft' if hemi is 'L' else 'CortexRight',
                    'Name':'Gradient {}'.format(g+1)
                    }
                )
        )
    return gii

# from neurohackacademy tutorial
def surf_data_from_cifti(data, surf_name):
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
    surf_name = 'CIFTI_STRUCTURE_'+surf_name.strip()
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")

