export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=8989 train.py \
    --config_path configs/sft_wan22_ic.yaml \
    --save_dir outputs/sft_wan22_ic
