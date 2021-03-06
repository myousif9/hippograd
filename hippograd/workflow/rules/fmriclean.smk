rule fmriclean:
    input: 
        bold_surf = config['input_path']['bold_surf'],
        bold_vol = config['input_path']['bold_volume'],
        mask = config['input_path']['mask']
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
    threads: 8
    resources:
        mem_mb = 16000,
        time = 60    
    group: 'subj'
    log: bids(root = 'logs',**bold_surf_wildcards, suffix = 'fmriclean.txt')
    script:
        '../scripts/fmriclean.py'