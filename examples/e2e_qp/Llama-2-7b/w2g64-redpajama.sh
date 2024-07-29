CUDA_VISIBLE_DEVICES=0 python main_e2e_qp.py \
    --quant_model_path ./output/block_ap_models/Llama-2-7b-w2g64 \
    --model_family Llama-2 \
    --wbits 2 \
    --group_size 64 \
    --learning_rate 2e-5 \
    --dataset redpajama \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/Llama-2-7b-w2g64-redpajama-4096 \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --save_strategy epoch \
    --training_strategy epochs \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --max_train_samples 4096 \
    --num_train_epochs 1 \
    --eval_dataset_size 64 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --preprocessing_num_workers 32 \
    --do_ppl_eval
