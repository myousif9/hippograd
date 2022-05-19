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
    group: 'subj'
    script:
        '../scripts/fmriclean.py'

# if config['atlas'] == None:
#     rule reduce_label:
#         input:
#             surf = rules.fmriclean.fmri_surf 
#         params:
#             template = 'fsLR32k',
#             atlas = confg['atlas'],
#             n_regions = config['n_regions'],
#             atlas_kwargs = config = config['atlas_kwargs']
#         output:
#             surf = bold(
#                 root = 'results',
#                 datatype = 'func',
#                 den = '91k',
#                 space = 'fsLR'
#                 desc = config['atlas']+config['n_regions'],
#                 suffix = 'bold_.dtseries.nii'
#                 **bold_surf_wildcards
#             )
#         run: 


## Add rule for masking regions