# 自定义超参数
export DIR_NAMAE=WSDM2022-RGCN_SAGE
export service=sing

export exp_name=node_feats_learnable
# export job_name=
export dim_node=128
export dim_time=10
export time_encoder_type=period
export embedding=RGCN_SAGE

# export data_name=
export lr=0.001
export learn_node_feats=True
export n_epoch=6
export n_layers=2
export weight_decay=5e-4

source keys.sh

for data_name in "A" "B" ; do
    export data_name=$data_name

    export job_name=${data_name}-node_feats_learnable

    if [[ ! -v upload_data ]]; then
        amlt run amulet_submit/submit.yaml $exp_name --upload-data -t msrresrchvc --sku G1-A100 --sla Basic
        export upload_data=True
    else
        amlt run amulet_submit/submit.yaml $exp_name -t msrresrchvc --sku G1-A100 --sla Basic
    fi
done
