/home/anconda3/envs/longcat-image2/bin/python run_cedit_longcat_inference.py \
    --pretrained_model "/home/disk2/hwj/my_hf_cache/hub/models--meituan-longcat--LongCat-Image-Edit/snapshots/7b54ef423aa7854be7861600024be5c56ab7875a" \
    --lora_weight "/home/disk2/hwj/LongCat-Image/output/edit_lora_model-stage2-final-0225/checkpoints-10000" \
    --model_name "my_cedit_model" \
    --output_root "results_cedit" \
    --lang both \
    --num_images 1 \
    --task_type "Background Change"