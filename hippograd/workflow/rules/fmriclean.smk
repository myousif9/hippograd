rule fmriclean:
    input: 
        bold_surf = rfmri = lambda wildcards: fmri_surf_dict[wildcards.subject],
        bold_vol =  rfmri = lambda wildcards: fmri_vol_dict[wildcards.subject]
    params:
        strategy = config['fmri_clean_strategy']
    output: 
        fmri_surf = bids(
            root = 'results',
            datatype = 'func',
            space = 'fsLR',
            den = '91k',
            desc = 'cleaned',
            suffix = 'bold.dtseries.nii',
            **subj_wildcards
        ),
        fmri_vol = bids(
            root = 'results',
            datatype = 'func',
            desc = 'cleaned',
            suffix = 'bold.dtseries.nii',
            **subj_wildcards
        )
    script:
        'scripts/fmriclean.py'
 
