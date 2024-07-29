import torch
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import random
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import random, torch

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from quantize.int_linear_real import load_quantized_model
import argparse

# python3 /auto/regrt/sw/dgundimeda/EfficientQAT/dequantize_effqat_weights.py --model /auto/regrt/sw/dgundimeda/qwen_models/qwen2_7b_llamafied --gptq_model /auto/regrt/sw/dgundimeda/EfficientQAT/output/block_ap_models/Qwen-1.5-7b-llama-w3g64_w2g64 --output_dir qwen_1_5_base_int3_g64_int2_g64_effqat_blockap_redpajama
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--gptq_model", default="Qwen/Qwen1.5-7B-Chat-GPTQ-Int4", type=str, help="model path for the gptq model")
parser.add_argument("--output_dir", default="./merged_gptq_weights", type=str, help="Path to save the merged model")
parser.add_argument("--before_eval_ppl",action = "store_true")
parser.add_argument("--after_eval_ppl",action = "store_true")
parser.add_argument("--bits",type=int,default=3, 
					choices=[2,3,4,6,8], help="Number of bits to quantize the model to")
parser.add_argument("--group_size",type=int,default=64, 
					choices=[-1,16,32,64,128], help="group size for the group quantization")

args = parser.parse_args()
import os
def create_folder(folder_name):
	current_path = os.getcwd()
	folder_path = os.path.join(current_path, folder_name)

	if not os.path.exists(folder_path):
		os.makedirs(folder_path, exist_ok=True)  # exist_ok avoids errors if it already exists
		print(f"Folder '{folder_name}' created successfully!")
	else:
		print(f"Folder '{folder_name}' already exists.")


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

import transformers

@torch.no_grad()
def evaluate(model):
	# model.to('cuda')
	tokenizer = transformers.AutoTokenizer.from_pretrained(
		pretrained_model_name_or_path=args.model,
		model_max_length=2048,
		padding_side="right",
		use_fast=False,
	)
	tokenizer.pad_token = tokenizer.eos_token


	def get_wikitext2(nsamples, seed, seqlen, tokenizer):
		print("get_wikitext2")
		traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
		testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

		# tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
		trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
		testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

		
		random.seed(seed)
		trainloader = []
		for _ in range(nsamples):
			i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
			j = i + seqlen
			inp = trainenc.input_ids[:, i:j]
			tar = inp.clone()
			tar[:, :-1] = -100
			trainloader.append((inp, tar))
		return trainloader, testenc

	nsamples = 128
	seqlen = 2048
	seed = 0

	dataloader, testloader = get_wikitext2(
		nsamples = nsamples,
		seed = seed,
		seqlen = seqlen,
		tokenizer = tokenizer,
	)
	
	testenc = testloader.input_ids

	# use_cache = model.config.use_cache
	model.config.use_cache = False
	model.eval()
	nlls = []
	nsamples = testenc.numel() // seqlen
	for i in tqdm(range(nsamples)):
		with torch.no_grad():
			batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
			logits = model(batch)['logits'].to('cpu')
			# batch.to('cpu')
			shift_logits = logits[:, :-1, :]
			shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(
				shift_logits.view(-1, shift_logits.size(-1)),
				shift_labels.view(-1),
			)
			neg_log_likelihood = loss.float() * seqlen
			nlls.append(neg_log_likelihood)
			del batch, logits, shift_logits, shift_labels
			torch.cuda.empty_cache()

	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
	print("perplexity: ", ppl.item())


create_folder(args.output_dir)
# model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
# if args.before_eval_ppl:
# 	print("Calculating perplexity before replacing weights...")
# 	evaluate(model)
	
# del tokenizer
# exit()
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", trust_remote_code=True)
# gptq_model = AutoModelForCausalLM.from_pretrained(args.gptq_model, device_map="auto", trust_remote_code=True)
gptq_model, gptq_tokenizer = load_quantized_model(args.gptq_model,args.bits, args.group_size)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)                           
from tqdm import tqdm
 
for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].mlp.down_proj.scales
	qweight = gptq_model.model.layers[i].mlp.down_proj.qweight
	qzeros = gptq_model.model.layers[i].mlp.down_proj.qzeros
	g_idx = gptq_model.model.layers[i].mlp.down_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].mlp.down_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()

for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].mlp.up_proj.scales
	qweight = gptq_model.model.layers[i].mlp.up_proj.qweight
	qzeros = gptq_model.model.layers[i].mlp.up_proj.qzeros
	g_idx = gptq_model.model.layers[i].mlp.up_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].mlp.up_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()
 
for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].mlp.gate_proj.scales
	qweight = gptq_model.model.layers[i].mlp.gate_proj.qweight
	qzeros = gptq_model.model.layers[i].mlp.gate_proj.qzeros
	g_idx = gptq_model.model.layers[i].mlp.gate_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].mlp.gate_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()
 

for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].self_attn.q_proj.scales
	qweight = gptq_model.model.layers[i].self_attn.q_proj.qweight
	qzeros = gptq_model.model.layers[i].self_attn.q_proj.qzeros
	g_idx = gptq_model.model.layers[i].self_attn.q_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].self_attn.q_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()
 
for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].self_attn.k_proj.scales
	qweight = gptq_model.model.layers[i].self_attn.k_proj.qweight
	qzeros = gptq_model.model.layers[i].self_attn.k_proj.qzeros
	g_idx = gptq_model.model.layers[i].self_attn.k_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].self_attn.k_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()
 
for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].self_attn.v_proj.scales
	qweight = gptq_model.model.layers[i].self_attn.v_proj.qweight
	qzeros = gptq_model.model.layers[i].self_attn.v_proj.qzeros
	g_idx = gptq_model.model.layers[i].self_attn.v_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].self_attn.v_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()
 
for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].self_attn.o_proj.scales
	qweight = gptq_model.model.layers[i].self_attn.o_proj.qweight
	qzeros = gptq_model.model.layers[i].self_attn.o_proj.qzeros
	g_idx = gptq_model.model.layers[i].self_attn.o_proj.g_idx
	w_bits = 2 if i in [22,23,24,25,26,27] else 3
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,w_bits, args.group_size)
	model.model.layers[i].self_attn.o_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()

del gptq_model
model.to(torch.bfloat16)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
torch.cuda.empty_cache()

# model.to('cuda')
if args.after_eval_ppl:
	print("Calculating perplexity after replacing weights...")
	evaluate(model)
