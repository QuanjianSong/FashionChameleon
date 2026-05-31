export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=1 python predictor/stream_infer_ic.py \
    --config_path configs/gr_dmd_wan22_ic.yaml \
    --seed 42 \
    --h 1280 \
    --w 704 \
    --num_frames 81 \
    --output_path samples/gr_dmd_wan22_ic_reward/ \
    --checkpoint XXX
