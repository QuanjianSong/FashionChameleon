export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=7777 train.py \
    --config_path configs/tf_wan22_ic.yaml \
    --save_dir outputs/tf_wan22_ic
