export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=1 python causal_infer_ic.py --config_path configs/tf_wan22_ic.yaml \
    --seed 42 \
    --h 1280 \
    --w 704 \
    --num_frames 81 \
    --output_path samples/tf_wan22_ic/ \
    --checkpoint XXX
