export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 --master_port=1234 trainer/train.py \
    --config_path configs/gr_dmd_wan22_ic.yaml \
    --save_dir outputs/gr_dmd_wan22_ic
