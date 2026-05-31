export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=1 python predictor/infer_ic.py \
    --config_path configs/sft_wan22_ic.yaml \
    --seed 42 \
    --h 1280 \
    --w 704 \
    --num_frames 81 \
    --output_path samples/sft_wan22_ic/ \
    --checkpoint XXX
