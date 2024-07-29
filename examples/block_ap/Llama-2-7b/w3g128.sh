CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--model /auto/regrt/sw/dgundimeda/qwen_models/qwen1_to_qwen2llama   \
--output_dir ./output/block_ap_log/Llama-2-7b-w3g128 \
--net Llama-2 \
--wbits 3 \
--group_size 128 \
--quant_lr 1e-4 \
--weight_lr 1e-5 \
--real_quant \
--eval_ppl \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--save_quant_dir ./output/block_ap_models/Llama-2-7b-w3g128