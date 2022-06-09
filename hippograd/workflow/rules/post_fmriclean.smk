from scripts.utilities import fetch_atlas_path
from os.path import join

bold_surf_clean = bids(
    root = 'results',
    datatype = 'func',
    den = '91k',
    space = 'fsLR',
    desc = 'cleaned',
    suffix = 'bold.dtseries.nii',
    **bold_surf_wildcards
)

bold_vol_clean = bids(
    root = 'results',
    datatype = 'func',
    space = 'MNI152NLin2009cAsym',
    desc = 'cleaned',
    suffix = 'bold.nii.gz',
    **bold_vol_wildcards
)

fmriclean_vol_dict = dict(zip(bold_vol_ziplist['subject'],  expand(bold_vol_clean,zip,**bold_vol_ziplist)))
fmriclean_surf_dict = dict(zip(bold_surf_ziplist['subject'], expand(bold_surf_clean,zip,**bold_surf_ziplist)))

rule map_rfmri_hippunfold_surface:
    input:
        check_struct = rules.set_surf_structure.output.check,
        surf = rules.csv2gifti.output.surf,
        fmri_vol = lambda wildcards: fmriclean_vol_dict[wildcards.subject]
    params:
        structure = 'CORTEX_LEFT' if '{hemi}' == 'L' else 'CORTEX_RIGHT'
    output:
        fmri_surf = bids(
            root = 'results',
            datatype = 'func',
            task =  '{task}',
            hemi = '{hemi}',
            space = 'MNI152NLin2009cAsym',
            den = '{density}',
            suffix = 'bold.func.gii',
            **subj_wildcards
            ),
    container: config['singularity']['autotop']
    group: 'subj'
    threads: 8
    resources:
        mem_mb = 16000,
        time = 60    
    log: bids(root = 'logs',**subj_wildcards, task = '{task}', hemi = '{hemi}', den = '{density}', suffix = 'map-rfmri-hippunfold-surface.txt')
    shell:
        '''
        wb_command -volume-to-surface-mapping {input.fmri_vol} {input.surf} {output.fmri_surf} -trilinear &> {log}
        '''

rule compute_gradients:
    input:
        rfmri_hipp = rules.map_rfmri_hippunfold_surface.output.fmri_surf,
        rfmri_ctx = lambda wildcards: fmriclean_surf_dict[wildcards.subject]
    params:
        n_gradients = config['n_gradients'],
        parcellation = fetch_atlas_path(config['cortex_parcellation'],config['n_parcel'], join(workflow.basedir,'..','resources','parcellations')),
        refgradL = join(workflow.basedir,'..',config['reference_gradient'][0]),
        refgradR = join(workflow.basedir,'..',config['reference_gradient'][1]),
        kernel = config['affinity_kernel'],
        embedding = config['embedding_approach'],
        align = config['align_method'],
        density = config['density'],
    output:
        correlation_matrix = bids(
            root = 'results',
            datatype = 'func',
            task = '{task}',
            hemi = '{hemi}',
            space = 'MNI152NLin2009cAsym',
            den = '{density}',
            suffix = 'correlationmatrix.npy',
            **subj_wildcards
            ),
        lambdas = bids(
            root = 'results',
            datatype = 'func',
            task = '{task}',
            hemi = '{hemi}',
            space = 'MNI152NLin2009cAsym',
            den = '{density}',
            suffix = 'lambdas.npy',
            **subj_wildcards
            ),
        gradient_maps = bids(
            root = 'results',
            datatype = 'func',
            task = '{task}',
            hemi = '{hemi}',
            space = 'MNI152NLin2009cAsym',
            den = '{density}',
            desc = 'aligned',
            suffix = 'gradients.func.gii',
            **subj_wildcards
            ),
    threads: 8
    resources:
        mem_mb = 16000,
        time = 60
    group: 'subj'
    log: bids(root = 'logs',**subj_wildcards, task = '{task}', hemi = '{hemi}', den = '{density}', suffix = 'compute_gradients.txt')
    script: '../scripts/compute_gradients.py'

rule grad_smooth:
    input:
        gradmap = rules.compute_gradients.output.gradient_maps,
        surf = rules.csv2gifti.output.surf
    params:
        structure = 'CORTEX_LEFT' if '{hemi}' == 'L' else 'CORTEX_RIGHT',
        smoothing_kernel = config['smoothing_kernel']
    output:
        check = bids(
            root = 'work',
            datatype = 'func',
            hemi = '{hemi}',
            task = '{task}',
            den = '{density}',
            suffix = 'calc_grad.done',
            **subj_wildcards
        ),
        surf_smooth = bids(
            root = 'results',
            datatype = 'func',
            task = '{task}',
            hemi = '{hemi}',
            space = 'MNI152NLin2009cAsym',
            den = '{density}',
            desc = 'smoothed',
            suffix = 'gradients.func.gii',
            **subj_wildcards
            ),
    container: config['singularity']['autotop']
    group: 'subj'
    log: bids(root = 'logs',**subj_wildcards, task = '{task}', hemi = '{hemi}', den = '{density}', suffix = 'grad_smooth.txt')
    shell: 
        '''
        wb_command -set-structure {input.gradmap} {params.structure} -surface-type ANATOMICAL &> {log}
        wb_command -metric-smoothing {input.surf} {input.gradmap} {params.smoothing_kernel} {output.surf_smooth}
        touch {output.check}
        '''
rule create_spec:
    input: 
        surf = rules.csv2gifti.output.surf,
        rfmri_hipp = rules.map_rfmri_hippunfold_surface.output.fmri_surf, 
        gradmap = rules.compute_gradients.output.gradient_maps,
        gradmap_smooth = rules.grad_smooth.output.surf_smooth,
    params:
        structure = 'CORTEX_LEFT' if '{hemi}' == 'L' else 'CORTEX_RIGHT',
        unfold = join(workflow.basedir,'..',config['reference_gradient'][0])
    output:
        spec = bids(
            root = 'work',
            task = '{task}',
            hemi = '{hemi}',
            den = '{density}',
            suffix = 'hippograd.spec',
            **subj_wildcards
        ),
    container: config['singularity']['autotop']
    group: 'subj' 
    shell: 
        '''
        wb_command -add-to-spec-file {output.spec} {params.structure} {input.surf}
        wb_command -add-to-spec-file {output.spec} {params.structure} {input.rfmri_hipp}
        wb_command -add-to-spec-file {output.spec} {params.structure} {input.gradmap}
        wb_command -add-to-spec-file {output.spec} {params.structure} {input.gradmap_smooth}
        ''' 

rule joinLRspec:
    input: 
        spec = expand(rules.create_spec.output.spec, hemi = config['hemi'], allow_missing = True)
    output: 
        spec = bids(
            root = 'results',
            task = '{task}',
            den = '{density}',
            suffix = 'hippograd.spec',
            **subj_wildcards
        ),
    container: config['singularity']['autotop']
    group: 'subj'
    shell:
        '''
        wb_command -spec-file-merge {input.spec} {output.spec}
        ''' 