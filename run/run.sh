if [[ ! -v AMLT_JOB_NAME ]]; then
    source /anaconda/bin/activate
    conda activate py38_pytorch
fi

# remeber: `conda activate` will remove pythonpath
PROJECT_DIR=$(readlink -f $PWD/..)
source ../export_env.sh

python src/main.py \
    --exp_name $exp_name \
    --job_name $job_name \
    --dim_node $dim_node \
    --dim_time $dim_time \
    --time_encoder_type $time_encoder_type \
    --embedding $embedding \
    --data_name $data_name \
    --lr $lr \
    --n_epoch $n_epoch \
    --n_layers $n_layers \
    --weight_decay $weight_decay

