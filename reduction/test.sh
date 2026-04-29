#!/bin/bash

host=`hostname -s`
echo $CUDA_VISIBLE_DEVICES



# python experiment.py \
#        --experiment_type "reduction_dpgmm_e3_theia"

# python experiment.py \
#        --experiment_type "reduction_dpgmm_e3_clearscope"

# python experiment.py \
#        --experiment_type "reduction_dpgmm_e3_cadets"

# python experiment.py \
#        --experiment_type "reduction_dpgmm_e5_theia" \


# python experiment.py \
#        --experiment_type "reduction_dbstream_e3_theia" \


# python experiment.py \
#        --experiment_type "reduction_dpgmm_monday_cic_ids" \

# python experiment.py \
#        --experiment_type "reduction_truncation_tradeoff"


# python experiment.py \
#        --experiment_type "reduction_sampling_tradeoff"

python experiment.py \
       --experiment_type "update_fidelity"