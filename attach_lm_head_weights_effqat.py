from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
import argparse
import torch
from quantize.int_linear_real import load_quantized_model

# python3 /auto/regrt/sw/dgundimeda/EfficientQAT/dequantize_effqat_weights.py --model /auto/regrt/sw/dgundimeda/qwen_models/qwen1_to_qwen2llama --gptq_model /auto/regrt/sw/dgundimeda/EfficientQAT/output/block_ap_models/Llama-2-7b-w3g64_w2g64 --output_dir qwen_1_0_base_int3_g64_int2_g64_effqat_blockap
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")

args = parser.parse_args()
output_dir = args.model + "_lm_head_attach"

def dequantization(qweight, qzeros, scales, g_idx,bits=4, group_size=128):
	# Create a tensor for bitwise right shift operation
	dim_2 = scales.shape[0]*group_size
	wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0).to(qzeros.device)
	# Apply bitwise right shift and convert qzeros to the appropriate type
	import math
	zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, math.ceil(32 / bits)), wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
	torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros) #zeros (86,512,8)
	zeros=zeros[:,:,:32//bits]
	# zeros = zeros + 1
	zeros = zeros.reshape((zeros.shape[0],zeros.shape[1]*zeros.shape[2])) # zeros (86,1,4096)

	zeros = zeros[:,:scales.shape[-1]]
 
	# Reshape the scales tensor
	# scales = scales.reshape(-1, 1, scales.shape[-1]) # scales (86,1,4096)

	# Similar bitwise right shift operation for qweight and reshape
	# breakpoint()
	weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, math.ceil(32 / bits), -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8) # weight (1376,8,4096)
	torch.bitwise_and(weight, (2 ** bits) - 1, out=weight) # weight (1376,8,4096)
	weight = weight[:,:32//bits,:]
	# weight = weight.reshape(-1, group_size, weight.shape[2]) # weight (86,128,4096)

	# Apply dequantization formula and reshape the final weight
	
	weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
	weight = weight[:dim_2,:]
	weight = scales[g_idx] * (weight - zeros[g_idx])
	# Return the transposed weight
	return weight.transpose(0, 1)	

model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", trust_remote_code=True)
# gptq_model = AutoModelForCausalLM.from_pretrained(args.gptq_model, device_map="auto", trust_remote_code=True)
gptq_model, gptq_tokenizer = load_quantized_model(args.gptq_model,args.bits, args.group_size)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)                           
breakpoint()
scales = gptq_model.model.lm_head.scales
qweight = gptq_model.model.lm_head.qweight
qzeros = gptq_model.model.lm_head.qzeros
g_idx = gptq_model.model.lm_head.g_idx
w_bits = 4
dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
model.model.lm_head.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())

model.to(torch.bfloat16)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
torch.cuda.empty_cache()