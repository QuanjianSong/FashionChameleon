export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1234 trainer/train.py \
    --config_path configs/tf_wan22_ic.yaml \
    --save_dir outputs/tf_wan22_ic
