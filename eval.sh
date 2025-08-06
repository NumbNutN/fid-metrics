
# for pretrained
python run_metrics.py \
    --gen_data_root video_data_gen_pretrained \
    --log_file metrics_log_pretrained.jsonl \
    --ip_address 172.16.204.187 \
    --ports 23991


# for finetune
python run_metrics.py \
    --gen_data_root video_data_gen_finetune \
    --log_file metrics_log_finetune.jsonl \
    --ip_address 172.16.204.187 \
    --ports 24001