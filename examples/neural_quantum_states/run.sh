#!/bin/bash

MOLECULE_LIST=('H2O' 'N2')
BASIS_LIST=('sto-3g')
for MOLECULE in "${MOLECULE_LIST[@]}"
do
    for BASIS in "${BASIS_LIST[@]}"
    do
	LWS_VAR=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        args=(
            --config_file './exp_configs/nqs.yaml' DDP.WORLD_SIZE $LWS_VAR DDP.NODE_IDX 0 DDP.LOCAL_WORLD_SIZE $LWS_VAR
	    MODEL.MODEL_NAME 'retnet' DATA.MOLECULE "${MOLECULE}" DATA.BASIS "${BASIS}" 
        )
	CUDA_VISIBLE_DEVICES=$(seq 0 $((LWS_VAR - 1)) | tr '\n' ',') python -m main "${args[@]}"
    done
done
