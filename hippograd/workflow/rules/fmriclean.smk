rule fmriclean:
    input: 
        bold_surf = config['input_path']['bold_surf'],
        bold_vol = config['input_path']['bold_volume']
    params:
        strategy = config['fmri_clean_strategy']
    output: 
        fmri_surf = bids(
            root = 'results',
            datatype = 'func',
            den = '91k',
            space = 'fsLR',
            desc = 'cleaned',
            suffix = 'bold.dtseries.nii',
            **bold_surf_wildcards
        ),
        fmri_volume = bids(
            root = 'results',
            datatype = 'func',
            space = 'MNI152NLin2009cAsym',
            desc = 'cleaned',
            suffix = 'bold.nii.gz',
            **bold_vol_wildcards
        )
    group: 'subj'
    script:
        'scripts/fmriclean.py'