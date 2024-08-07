from transformers import AutoTokenizer , AutoModelForCausalLM
import torch, json, os
import functools, copy
from datetime import datetime
import torch.nn as nn
from datautils import get_loaders
from tqdm import tqdm
import warnings
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
warnings.filterwarnings("ignore")
from transformers import logging

# logging.set_verbosity_error()

import argparse, random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
all_outputs = ['embed_tokens', 'input_layernorm', 'post_attention_layernorm', 'norm', 'lm_head'
			   'q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkt_proj', 'sfmx' , 'attn_output_layer', 'rotary_emb',
			   'gate_proj', 'up_proj', 'down_proj' , 'act_fn', 'eltmul',
			   'attn_residue', 'mlp_residue']
outputs_8bit = ['q_proj','k_proj','v_proj','qkt_proj','attn_output_layer','gate_proj' ,'act_fn','up_proj','norm', 'rotary_pos_emb']
outputs_16bit = ['down_proj','embed_tokens','attn_residue','mlp_residue','eltmul','sfmx','o_proj','lm_head']
patterns = outputs_8bit + outputs_16bit

def round_ste(x: torch.Tensor):
	return (x.round() - x).detach() + x


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--calib_dataset",type=str,default="wikitext2", 
					choices=["wikitext2", "ptb", "c4", "mix","pile"], help="Where to extract calibration data from.")
parser.add_argument("--quantize_weights",action = "store_true")
parser.add_argument("--moving_avg",action = "store_true")
parser.add_argument("--gptq",action = "store_true")


args = parser.parse_args()

# Load the Model and Tokenizer
if args.gptq:
	# model = AutoGPTQForCausalLM.from_quantized(args.model, device="cuda:0")
	pass
else:
	model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",torch_dtype = "auto", trust_remote_code=True)

if args.gptq:
	tokenizer = AutoTokenizer.from_pretrained("/auto/regrt/sw/dgundimeda/qwen_models/qwen2_7b_llamafied",trust_remote_code=True)
else:
	tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
print("Loaded model into device:", model.device)

# from hqq.core.quantize import *

# dummy_lm_head = torch.nn.Linear(1, 1, bias=False)
# dummy_lm_head.weight.data = model.lm_head.weight
# dev = model.lm_head.weight.device
# quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
# hqq_layer  = HQQLinear(dummy_lm_head, quant_config=quant_config, compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
# model.lm_head.weight = torch.nn.Parameter(hqq_layer.dequantize().to(dev))

def qdq(tensor, n_bits = 8 ,group_size=None,offset_enable=False):
	tensor_shape = tensor.shape

	if group_size is not None:
		tensor = tensor.reshape(-1, group_size)

	reduce_shape = [-1]
	xmin = tensor.amin(reduce_shape, keepdim=True)
	xmax =  tensor.amax(reduce_shape, keepdim=True)
	if offset_enable:
		diff = xmax - xmin
		scale = diff/(2**(n_bits)-1)
		scale = scale.clamp(min=1e-6, max=1e6)
		offset = round_ste(-xmin/scale)
		tensor_int = round_ste(tensor / scale)
		tensor_int = tensor_int.add(offset)
		tensor_int = torch.clamp(tensor_int, 0, (2**(n_bits))-1)
		tensor_int = tensor_int.sub(offset)
		tensor_dequant = tensor_int.mul(scale)
	else:
		abs_max = torch.max(xmax.abs(),xmin.abs())
		scale = abs_max / (2**(n_bits-1)-1)
		scale = scale.clamp(min=1e-6, max=1e6)
		tensor_int = torch.clamp(round_ste(tensor / scale) , -2**(n_bits-1), (2**(n_bits-1))-1)
		tensor_dequant = tensor_int.mul(scale)
	if group_size is not None:
		tensor_dequant = tensor_dequant.reshape(tensor_shape)
	return tensor_dequant

def quantize_model_weights_config(model, default_bits = 4, default_group_size=64, offset_enable = True):
	config = model.config.to_dict()
	quant_strategy = config["quant_strategy"]
	for name,module in model.named_modules():
		if isinstance(module, nn.Linear):
			if "lm_head" in name:
				module.weight = torch.nn.Parameter(qdq(module.weight, default_bits, default_group_size, offset_enable))
				continue
			layer_idx, layer_name = name.split(".")[2], name.split(".")[-1]
			if layer_idx in quant_strategy:
				decoder_layer_strategy = quant_strategy[layer_idx]
				bits = decoder_layer_strategy[layer_name+"-n_bits"] if layer_name+"-n_bits" in decoder_layer_strategy else default_bits
				group_size = decoder_layer_strategy[layer_name+"-group_size"] if layer_name+"-group_size" in decoder_layer_strategy else default_group_size
			module.weight = torch.nn.Parameter(qdq(module.weight,bits, group_size, offset_enable))
			print(f"Quantizing wts of layer: {name}, bits: {bits} group: {group_size}")
	return model


def quantize_model_weights(model, n_bits = 8, group_size = None):
	for name,module in model.named_modules():
		if isinstance(module, nn.Linear):
			patterns_2 = [f".{i}." for i in [22,23,24,25,26,29]]
			if "lm_head" in name:
				if 'weight' in [name1 for name1, param in module.named_parameters()]:
					# print(f"Quantizing the lm_head weight of the Layer:{name}, bits: 4, group_size: 64")
					module.weight = torch.nn.Parameter(qdq(module.weight,4, 64, True))
			# elif "down_proj" in name:
			# 	patterns_4 = [f".{i}." for i in [2,7,8,15,28,29,30,31]]#[2,8,30,31], 
			# 	flag = False
			# 	for pattern in patterns_4:
			# 		if pattern in name:
			# 			print(f"Quantizing the dp weight of the Layer:{name}, bits: 4, group_size: 64")
			# 			module.weight = torch.nn.Parameter(qdq(module.weight,4, 64,True))
			# 			flag = True
			# 			continue
			# 	if not flag:
			# 		print(f"Quantizing the dp weight of the Layer:{name}, bits: 3, group_size: 64", module.weight.shape)
			# 		module.weight = torch.nn.Parameter(qdq(module.weight,3, 64, True))
			# elif "gate_proj" in name:
			# 	patterns_2 = [f".{i}." for i in [21,22,23,25,26,27]]#[2,8,30,31]
			# 	flag = False
			# 	for pattern in patterns_2:
			# 		if pattern in name:
			# 			print(f"Quantizing the dp weight of the Layer:{name}, bits: 4, group_size: 64")
			# 			module.weight = torch.nn.Parameter(qdq(module.weight,2, 64,True))
			# 			flag = True
			# 			continue
			# 	if not flag:
			# 		print(f"Quantizing the dp weight of the Layer:{name}, bits: 3, group_size: 64", module.weight.shape)
			# 		module.weight = torch.nn.Parameter(qdq(module.weight,3, 64, True))
			else:
				flag = False
				patterns_2 = [f".{i}." for i in [22,23,24,25,26,27]]
				for pattern in patterns_2:
					if pattern in name:
						# print(f"Quantizing the dp weight of the Layer:{name}, bits: 2, group_size: 64", module.weight.shape)
						module.weight = torch.nn.Parameter(qdq(module.weight,2, 64, True))
						flag = True
						continue
				if not flag:
					# print(f"Quantizing the dp weight of the Layer:{name}, bits: 3, group_size: 64", module.weight.shape)
					module.weight = torch.nn.Parameter(qdq(module.weight,3, 64, True))
   
		# elif isinstance(module, nn.Linear):
		# 	patterns_2 = [f".{i}." for i in range(22,28)]
		# 	for pattern in patterns_2:
		# 		if pattern in name:
		# 			print(f"Quantizing the weight of the Layer:{name}, bits: 2, group_size: 64")
		# 			module.weight = torch.nn.Parameter(qdq(module.weight,2, 64))
		# 			continue
				
			
	return model

def calibrate_model(model, dataset,samples_count=146):
	# model.to(device)
	print("Model in device: ", model.device)
	seqlen = 2048
	cache_testloader = os.path.join(os.getcwd(), f'cache/dataloader_Llama-2_redpajama_1024_64_2048_train.cache')
	if os.path.exists(cache_testloader): 
		testloader = torch.load(cache_testloader)
		print(f"load calibration from {cache_testloader}")
	else:
		dataloader, testloader = get_loaders(
					dataset,
					seed=2,
					model=args.model,
					seqlen=seqlen,
				)
		torch.save(testloader, cache_testloader)
	if "c4" in dataset:
		testenc = testloader
	else:
		testenc = testloader.input_ids
	
	nsamples = min(samples_count,testenc.numel() // seqlen)
	# nsamples = testenc.numel() // seqlen
	use_cache = model.config.use_cache
	model.config.use_cache = False
	model.eval()
	nlls = []
	for i in tqdm(range(nsamples)):
		with torch.no_grad():
			batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]

			logits = model(batch.to(model.device))['logits']
			shift_logits = logits[:, :-1, :].to('cpu')
			shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
							:, 1:].to('cpu')
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
	print(f'{dataset} : {ppl.item()}')
	model.config.use_cache = use_cache

def check_pattern(layer, patterns):
	for pattern in patterns:
		if pattern in layer:
			return True
	return False

def check_pattern(layer, patterns):
	for pattern in patterns:
		if pattern in layer:
			return True
	return False

def calculate_scale_offsets(min_max_dict, n_bits = 8 ,offset_enable = False):
	scale_offset_dict = {}
	for layer in min_max_dict:
		n_bits = 16 if check_pattern(layer, outputs_16bit) else 8
		scale_offset_dict[layer] = {}
		abs_max = max(map(abs, min_max_dict[layer]))
		scale_offset_dict[layer]['offset'] = 0 if offset_enable else 0
		scale_offset_dict[layer]['scale'] = (2**(n_bits-1) - 1)/abs_max if 'sfmx' not in layer else (2**(n_bits) - 1)/abs_max
		scale_offset_dict[layer]['bits'] = n_bits
		scale_offset_dict[layer]['sign'] = True if 'sfmx' not in layer else False
	return scale_offset_dict  

# Function: Provide result of prompt inference using the provided model & tokenizer
def prompt_inference(model, tokenizer, prompt):
  inputs = tokenizer(prompt , return_tensors="pt")
#   generate_ids = model.generate(input_ids = inputs.input_ids.to('cuda'),do_sample=True,temperature=0.85,top_k=3,top_p=0.95,repetition_penalty=1.2,max_new_tokens = 100)
  generate_ids = model.generate(input_ids = inputs.input_ids.to('cuda'))
# do_sample=True,top_k=2,top_p=0.95, temperature=0.9, num_return_sequences=3
#   Top k =40, top p =0.9 temperature=0.2
  #,do_sample=True,top_k=2,top_p=0.9, temperature=0.9, num_return_sequences=3
  result_prompt = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
  return result_prompt

def dump_samples_stats(model):
	model.eval()

	output_min_max = {}
	final_output_min_max = {}
	def stat_tensor(name, input_tensor, output_tensor, level = "tensor"):
		if isinstance(output_tensor, tuple):
			output_tensor = output_tensor[0]
		if level == "tensor":
			output_tensor1 = torch.flatten(output_tensor).detach()
		elif level == "channel":
			hidden_dim = input_tensor.shape[-1]
			output_tensor1 = output_tensor.view(-1, output_tensor.shape[-1]).detach()

		going_max = torch.max(output_tensor1, dim=0)[0].float().cpu()
		going_min = torch.min(output_tensor1, dim=0)[0].float().cpu()

		output_min_max[name] = [going_min, going_max]

		final_output_min_max[name] = [torch.min(final_output_min_max[name][0], going_min), 
									 torch.max(final_output_min_max[name][1], going_max)] if name in final_output_min_max else [going_min, going_max]
		
	def stat_tensor_mov_avg(name, input_tensor, output_tensor, level = "tensor"):
		if isinstance(output_tensor, tuple):
			output_tensor = output_tensor[0]
		if level == "tensor":
			output_tensor1 = torch.flatten(output_tensor).detach()
		elif level == "channel":
			hidden_dim = input_tensor.shape[-1]
			output_tensor1 = output_tensor.view(-1, output_tensor.shape[-1]).detach()

		going_max = output_tensor1.max().item()
		going_min = output_tensor1.min().item()

		output_min_max[name] = [going_min, going_max]
		averaging_constant = 0.9
		if name in final_output_min_max:
			min_val = (1 - averaging_constant) * final_output_min_max[name][0]+ averaging_constant * going_min
			max_val = (1 - averaging_constant) * final_output_min_max[name][1]+ averaging_constant * going_max
			final_output_min_max[name] = [min_val, max_val]
		else:
			final_output_min_max[name] = [going_min, going_max]
		

	def stat_input_hook(m, x, y, name):
		if args.moving_avg:
			stat_tensor_mov_avg(name, x,y)
		else:
			stat_tensor(name, x, y)

	print("Adding stats hooks..")
	hooks = []
	for name, m in model.named_modules():
		# if len(list(m.named_children())) == 0 and not name.endswith("emb"):
		if check_pattern(name, patterns):
			hooks.append(
				m.register_forward_hook(
				functools.partial(stat_input_hook, name=name))
			)
		else:
			pass
	print("Calibrating the model")
	calibrate_model(model, args.calib_dataset,10)
	
	for h in hooks:
		h.remove()
	print("Removed stats hooks..")
	return final_output_min_max

# Function to dump data in the file path
def dump_json(path, data):
	for key in data.keys():
		data[key][0] = data[key][0].item()
		data[key][1] = data[key][1].item()
	with open(path, 'w') as json_file:
		json.dump(data, json_file)

# Function to read json file and return the dict
def read_json(path):
	f = open(path)
	data = json.load(f)
	f.close
	return data


# Function to add hooks to the model using input_scale_offsets and output scale offsets
def add_qdq_hooks(model, input_scale_offsets = None, output_scale_offsets = None):
	hooks = []
	
	def quantize_per_tensor_16bit(tensor, scale, offset):
		quantized_tensor = torch.zeros_like(tensor, dtype=torch.int16)
		quantized_tensor = (tensor * scale).round() + offset
		quantized_tensor = torch.clamp(quantized_tensor, -32768, 32767)  # Ensure values within int16 range
		return quantized_tensor
	
	def quantize_per_tensor_8bit(tensor, scale, offset):
		quantized_tensor = torch.zeros_like(tensor, dtype=torch.int8)
		quantized_tensor = (tensor * scale).round() + offset
		quantized_tensor = torch.clamp(quantized_tensor, -128, 127)  # Ensure values within int8 range
		return quantized_tensor
	
	def quantize_per_tensor_u8bit(tensor, scale, offset):
		quantized_tensor = torch.zeros_like(tensor, dtype=torch.uint8)
		quantized_tensor = (tensor * scale).round() + offset
		quantized_tensor = torch.clamp(quantized_tensor, 0, 255)  # Ensure values within uint8 range
		return quantized_tensor

	def quantize_per_tensor_u16bit(tensor, scale, offset):
		quantized_tensor = torch.zeros_like(tensor, dtype=torch.uint16)
		quantized_tensor = (tensor * scale).round() + offset
		quantized_tensor = torch.clamp(quantized_tensor, 0, 2**16-1)  # Ensure values within uint16 range
		return quantized_tensor

	def dequantize_(quantized_tensor, scale, offset):
		return (quantized_tensor - offset)/scale
		
	def output_qdq_hook(model, target_layer, scale_offset={"scale": 1, "offset": 0, "bits": 8}):
		def output_qdq_hook_tensor(module, input, output):
			scale, offset, bits, sign = scale_offset['scale'], scale_offset['offset'], scale_offset['bits'], scale_offset['sign'] 
			if isinstance(output, tuple) and len(output) == 2:
				output = (dequantize_(quantize_per_tensor_16bit(output[0], scale, offset), scale, offset),dequantize_(quantize_per_tensor_16bit(output[1], scale, offset), scale, offset))
			elif bits == 16:
				if sign:
					output = dequantize_(quantize_per_tensor_16bit(output, scale, offset), scale, offset)  
				else:
					output = dequantize_(quantize_per_tensor_u16bit(output, scale, offset), scale, offset)  
			elif bits == 8:
				if sign:
					output = dequantize_(quantize_per_tensor_8bit(output, scale, offset), scale, offset)
				else:
					output = dequantize_(quantize_per_tensor_u8bit(output, scale, offset), scale, offset)
			return output
		
		hooks.append(target_layer.register_forward_hook(output_qdq_hook_tensor))
			
	for name, m in model.named_modules():
		if name in output_scale_offsets and "residue" not in name:
			output_qdq_hook(model, m, output_scale_offsets[name])
			
	return model, hooks


# prompt = "What's the command to list files on Linux?"
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")
# prompt = "Write a short description of Large language Model"
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")
# prompt = "Write a code to compute factorial of 21 in python"
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")
# prompt = "What is the value of 1+2+4 ?"
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")
# chinese_prompt = "告诉我有关联想公司的信息"
chinese_prompt = "介绍一下大连这个城市"
print("Prompt: ", chinese_prompt)
result = prompt_inference(model, tokenizer, chinese_prompt)
print("Quantized Prompt Result: ", result)
print(50*"*")
# print(prompt_inference(model, tokenizer, "can you tell me about China in 1000 words \n"))
# print(50*"*")
# print(50*"*")
# print(prompt_inference(model, tokenizer, "can you tell me about Beijing in 1000 words \n"))
# exit()
# breakpoint()
# calibrate_model(model, args.calib_dataset)

output_stats = dump_samples_stats(model)
dump_json("/auto/worka/dgundimeda/llama2_onnx/Qwen1_effqat/output_stats.json",output_stats)
output_scale_offsets = calculate_scale_offsets(output_stats)
with open("/auto/worka/dgundimeda/llama2_onnx/Qwen1_effqat/output_scale_offsets.json", 'w') as json_file:
	json.dump(output_scale_offsets, json_file)
# exit()
# with open("/auto/worka/dgundimeda/llama2_onnx/Qwen/titash_scale_offsets_dp.json") as f:
# 	output_scale_offsets = json.load(f)
if args.quantize_weights:
	model = quantize_model_weights(model, 4, 64)
	print(50*"*")
	print("Post Quantizing weights: ")
	calibrate_model(model, args.calib_dataset,10)
	# breakpoint()
	print("Prompt: ", chinese_prompt)
	result = prompt_inference(model, tokenizer, chinese_prompt)
	print("Quantized Prompt Result: ", result)
	print(50*"*")
	# prompt = "What's the command to list files on Linux?"
	# print("Prompt: ", prompt)
	# result = prompt_inference(model, tokenizer, prompt)
	# print("Prompt Result: ", result)
	# prompt = "Write a short description of Large Language Model "
	# print("Prompt: ", prompt)
	# result = prompt_inference(model, tokenizer, prompt)
	# print("WT Quantized Prompt Result: ", result)
	# result = prompt_inference(model, tokenizer, "Is tomato a vegetable")
	# print("WT Quantized Prompt Result: ", result)
	# print(50*"*")
	# prompt = "Write a code in python to compute factorial of 21. "
	# print("Prompt: ", prompt)
	# result = prompt_inference(model, tokenizer, prompt)
	# print("WT Quantized Prompt Result: ", result)
	# print(50*"*")
	# prompt = "What is the value of 1+2+4 is"
	# print("Prompt: ", prompt)
	# result = prompt_inference(model, tokenizer, prompt)
	# print("Prompt Result: ", result)
	# print(50*"*")
	# print(prompt_inference(model, tokenizer, "can you tell me about China in 1000 words \n"))
	# print(prompt_inference(model, tokenizer, "value of 1+2 is "))
print(50*"*")
model ,qdq_hooks = add_qdq_hooks(model, None, output_scale_offsets)
# calibrate_model(model, args.calib_dataset)


print("Prompt: ", chinese_prompt)
result = prompt_inference(model, tokenizer, chinese_prompt)
print("Quantized Prompt Result: ", result)
print(50*"*")

# prompt = "Write a code in python to compute factorial of 21."
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Quantized Prompt Result: ", result)
# print(50*"*")

# prompt = "What is the value of 1+2+4 ?"

# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")

# prompt = "Explain Newtons third law of motion."
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")


# prompt = "What is the capital of China"
# print("Prompt: ", prompt)
# result = prompt_inference(model, tokenizer, prompt)
# print("Prompt Result: ", result)
# print(50*"*")
# print(prompt_inference(model, tokenizer, "can you tell me about China in 1000 words \n"))
# breakpoint()
for h in qdq_hooks:
	h.remove()



