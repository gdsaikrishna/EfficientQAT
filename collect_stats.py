import argparse, functools
import sys, torch, random, os, json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from pathlib import Path
from datautils import get_loaders
from tqdm import tqdm 
import torch.nn as nn


class LlamaQuantRMSNorm(nn.Module):
		def __init__(self, module, eps=1e-6):
				"""
				LlamaRMSNorm is equivalent to T5LayerNorm
				"""
				super().__init__()
				#self.weight = nn.Parameter(torch.ones(hidden_size))
				self.weight = module.weight
				self.variance_epsilon = eps
 
		def forward(self, hidden_states):
				input_dtype = hidden_states.dtype
				hidden_states = hidden_states.to(torch.float32)
				variance = hidden_states.pow(2).sum(-1, keepdim=True)
				variance = qdq_group(variance, bits = 16, group_size = None, offset_enabled = False, is_two_power = True)
				variance = variance.div(hidden_states.shape[-1])
				std_deviation = torch.rsqrt(variance + self.variance_epsilon)
				#breakpoint()
				std_deviation = qdq_group(std_deviation, bits = 16, group_size = None, offset_enabled = False, is_two_power = True)
				hidden_states = hidden_states * std_deviation
				return self.weight * hidden_states.to(input_dtype)


def replace_norm(model):
	for name, module in model.named_modules():
		if isinstance(module, LlamaDecoderLayer):
			# module = LlamaQuantRMSNorm(module)
			module.input_layernorm = LlamaQuantRMSNorm(module.input_layernorm)
			module.post_attention_layernorm = LlamaQuantRMSNorm(module.post_attention_layernorm)
		if isinstance(module, LlamaModel):
			module.norm = LlamaQuantRMSNorm(module.norm)
	return model


seqlen = 2048
seed = 10

def get_dtype_info( n_bits = 8, sign = True):
	dtype = torch.int8
	if sign :
		if n_bits == 8:
			dtype = torch.int8
		elif n_bits == 16:
			dtype = torch.int16
		elif n_bits == 32:
			dtype = torch.int32
		else:
			dtype = torch.int8
	else:
		if n_bits == 8:
			dtype = torch.uint8
		elif n_bits == 16:
			dtype = torch.int16
		elif n_bits == 32:
			dtype = torch.int32
		else:
			dtype = torch.uint8
	return dtype, torch.iinfo(dtype)

class BaseHook(object):
	def __init__(self):
		self.sign = {}
		self.n_bits = {}
	
	def __call__(self, name):
		self.sign[name] = True
		self.n_bits[name] = 8

class MinMaxHook(BaseHook):
	def __init__(self):
		super().__init__()
		self.min_vals = {}
		self.max_vals = {}

	def __call__(self, module, input, output, name):
		super().__call__(name)
		if isinstance(output, tuple) or name.endswith(".mlp") or isinstance(module, LlamaModel):
			pass
		else:
		# Extract the minimum and maximum values from the output tensor
			if name not in self.min_vals:
				self.min_vals[name] = output.min().item()
				self.max_vals[name] = output.max().item()
			else:
				self.min_vals[name] = min(self.min_vals[name], output.min().item())
				self.max_vals[name] = max(self.max_vals[name], output.max().item())

class AbsMinMaxHook(BaseHook):
	def __init__(self):
		super().__init__()
		self.min_vals = {}
		self.max_vals = {}

	def __call__(self, module, input, output, name):
		super().__call__(name)
		# Extract the minimum and maximum values from the output tensor
		if name not in self.min_vals:
			self.min_vals[name] = torch.abs(output).min().item()
			self.max_vals[name] = torch.abs(output).max().item()
		else:
			self.min_vals[name] = min(self.min_vals[name], torch.abs(output).min().item())
			self.max_vals[name] = max(self.max_vals[name], torch.abs(output).max().item())

class MovingAverageMinMaxHook(BaseHook):
	def __init__(self, averaging_constant = 0.1):
		super().__init__()
		self.averaging_constant = averaging_constant
		self.min_vals = {}
		self.max_vals = {}
	
	def __call__(self, module, input, output, name):
		super().__call__(name)
		if name not in self.min_vals:
			self.min_vals[name] = output.min().item()
			self.max_vals[name] = output.max().item()
		else:
			new_min = output.min().item()
			new_max = output.max().item()
			self.min_vals[name] = (1 - self.averaging_constant) * self.min_vals[name] + self.averaging_constant * new_min
			self.max_vals[name] = (1 - self.averaging_constant) * self.max_vals[name] + self.averaging_constant * new_max


def quantize(tensor, scale, qmin, qmax):
	quantized_tensor = torch.clamp((tensor * scale).round() , qmin , qmax)
	return quantized_tensor
	
def dequantize(tensor, scale):
	return tensor / scale

def override_dtype(hook):
	for layer, value in hook.min_vals.items():
		if "sfmx" in layer:
			hook.n_bits[layer] = 16
		if value  >= 0:
			hook.sign[layer] = False
			print(f"Updating the Layer {layer} to be Unsigned as the min value is {value}")
	return hook
 
def add_stat_hooks(model, observer):
	hooks = []
	hook = MinMaxHook()
	for name, module in model.named_modules():
		hooks.append(
			module.register_forward_hook(
				functools.partial(hook, name = name)
			)
		)

	return model, hooks, hook

def calculate_scale_offset(hook, observer):
	dic = {}
	value = None
	if observer == "min_max":
		for name, value in hook.max_vals.items():
			abs_max = max(abs(value), abs(hook.min_vals[name]))
			dtype, dtype_info = get_dtype_info(hook.n_bits[name], hook.sign[name])
			if name not in dic:
				dic[name] = {}
			dic[name]["max"] = abs_max
			dic[name]["scale"] = dtype_info.max / abs_max 
			dic[name]["offset"] = 0
			dic[name]["bits"] = dtype_info.bits
			dic[name]["qmax"] = dtype_info.max
			dic[name]["sign"] = hook.sign[name]

	return dic

def qdq(tensor, scale, n_bits = 8, sign = True):
	dtype, dtype_info = get_dtype_info(n_bits, sign)
	quantized_tensor = torch.zeros_like(tensor, dtype=dtype)
	quantized_tensor = torch.clamp((tensor * scale).round() , dtype_info.min , dtype_info.max)
	qdq_tensor = quantized_tensor / scale
	return qdq_tensor

def round_ste(x: torch.Tensor):
	return (x.round() - x).detach() + x

def qdq_group(tensor, bits = 8, group_size = None, offset_enabled = False, is_two_power = False):
	tensor_shape = tensor.shape

	if group_size is not None:
		tensor = tensor.reshape(-1, group_size)

	reduce_shape = [-1]
	xmin = tensor.amin(reduce_shape, keepdim=True)
	xmax =  tensor.amax(reduce_shape, keepdim=True)
	if is_two_power:
		if offset_enabled:
			diff = -1*(xmax + xmin)/2
			abs_max = ((xmax - xmin)/2)
			abs_max = abs_max + 1.22443e-15
			intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
			fracPart = (bits-1)*torch.ones_like(intPart) - intPart
			scale = (2**fracPart)
			scale = scale.pow(-1)
			scale = scale.clamp(min=1e-6, max=1e6)
			offset = round_ste(diff / scale)
			offset = torch.clamp(offset, -2**(bits-1), (2**(bits-1))-1)
			tensor_int = torch.clamp(round_ste(tensor / scale) + offset, -2**(bits-1), (2**(bits-1))-1)
			tensor_dequant = tensor_int - offset
			tensor_dequant = tensor_dequant.mul(scale)
		else:
			abs_max = torch.max(xmax.abs(),xmin.abs())
			abs_max = abs_max + 1.22443e-15
			intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
			fracPart = (bits-1)*torch.ones_like(intPart) - intPart
			scale = (2**fracPart)
			scale = scale.pow(-1)
			scale = scale.clamp(min=1e-6, max=1e6)
			tensor_int = torch.clamp(round_ste(tensor / scale) , -2**(bits-1), (2**(bits-1))-1)
			tensor_dequant = tensor_int.mul(scale)
	else:
		if offset_enabled:
			diff = xmax - xmin
			scale = diff/(2**(bits)-1)
			scale = scale.clamp(min=1e-6, max=1e6)
			offset = round_ste(-xmin/scale)
			tensor_int = round_ste(tensor / scale)
			tensor_int = tensor_int.add(offset)
			tensor_int = torch.clamp(tensor_int, 0, (2**(bits))-1)
			tensor_int = tensor_int.sub(offset)
			tensor_dequant = tensor_int.mul(scale)
		else:
			abs_max = torch.max(xmax.abs(),xmin.abs())
			scale = abs_max / (2**(bits-1) - 1)
			scale = scale.clamp(min=1e-6, max=1e6)
			tensor_int = torch.clamp(round_ste(tensor / scale) , -2**(bits-1), (2**(bits-1))-1)
			tensor_dequant = tensor_int.mul(scale)
	if group_size is not None:
		tensor_dequant = tensor_dequant.reshape(tensor_shape)
	return tensor_dequant

class OutputQDQ32_8_Hook(object):
	def __init__(self):
		self.q_bits = 32

	def __call__(self, module, input, output, dic, name):
		#int32_scale = ((2**31)-1)/dic["max"]
		#if "self_attn" in name or "layernorm" in name:
		#if "self_attn" in name:
		#if "input_layernorm" in name:
		# if "k_proj" in name or "v_proj" in name:
		# 	int8_scale = dic["scale"]
		# 	qdq_int8 = qdq(output, int8_scale, dic["bits"], dic["sign"])
		# else:
		int32_scale = 2**14
		qdq_int32 = qdq(output, int32_scale, 32, True)
		# if "qkt_proj" in name:
		# 	qdq_int8 = qdq_group(qdq_int32, bits = 8, group_size = None, offset_enabled = False, is_two_power = True)
		# else:
		qdq_int8 = qdq_group(qdq_int32, bits = 8, group_size = 64, offset_enabled = False, is_two_power = True)
		#qdq_int8_group = qdq_group(qdq_int32, 8)
		output = qdq_int8
		return output

def add_output_hooks(model, scales):
	output_qdq_hooks = []
	output_qdq_hook = OutputQDQ32_8_Hook()
	for name, module in model.named_modules():
		if name in scales:
			output_qdq_hooks.append(
				module.register_forward_hook(
					functools.partial(output_qdq_hook, dic = scales[name], name = name)))
	return model, output_qdq_hooks, output_qdq_hook

def get_data_loader(args):
	if args.cache_dir:
		Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
	cache_testloader = f'{args.cache_dir}/testloader_{args.calib_dataset}_all.cache'
	if os.path.exists(cache_testloader):
		testloader = torch.load(cache_testloader)
		print(f"load calibration from {cache_testloader}")
		return testloader
	else:
		dataloader, testloader = get_loaders(
			args.calib_dataset,
			nsamples=128,
			seed=seed,
			model=args.model,
			seqlen=seqlen,
			)
				
		torch.save(testloader, cache_testloader)
		return testloader
	return None

@torch.no_grad()
def evaluate(model, args):
	results = {}
	dataloader = get_data_loader(args)
	if "c4" in args.calib_dataset:
		data = dataloader
	else:
		data = dataloader.input_ids
	
	nsamples = data.numel() // seqlen
	if args.limit > 0:
		nsamples = min(nsamples, args.limit)
	model.eval()
	nlls = []
	for i in tqdm(range(nsamples)):
		with torch.no_grad():
			batch = data[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
			outputs = model.model(batch)
	
			hidden_states = outputs[0]
			logits = model.lm_head(hidden_states)
			shift_logits = logits[:, :-1, :]
			shift_labels = data[:, (i * seqlen) : ((i + 1) * seqlen)][
						:, 1:
					].to(model.lm_head.weight.device)
			loss_fct = torch.nn.CrossEntropyLoss()
			loss = loss_fct(
						shift_logits.view(-1, shift_logits.size(-1)),
						shift_labels.view(-1),
					)
			neg_log_likelihood = loss.float() * seqlen
			nlls.append(neg_log_likelihood)
			del outputs, batch, logits, shift_logits, shift_labels
			torch.cuda.empty_cache()

	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
	print(f'{args.calib_dataset} : {ppl.item()}')
	results[args.calib_dataset] = ppl.item()
	return results

def quantize_model_weights(model):
	for name,module in model.named_modules():
		if isinstance(module, torch.nn.Linear):
			if "lm_head" in name:
				if 'weight' in [name1 for name1, param in module.named_parameters()]:
					print(f"Quantizing the lm_head weight of the Layer:{name}, bits: 4, group_size: 64", module.weight.shape)
					module.weight = torch.nn.Parameter(qdq_group(module.weight,4, 64, True))
			# elif "down_proj" in name:
			# 	pass
			# 	patterns_3 = [f".{i}." for i in [2,8,30,31]]#[2,7,8,15,28,29,30,31]
			# 	flag = False
			# 	for pattern in patterns_3:
			# 		if pattern in name:
			# 			print(f"Quantizing the dp weight of the Layer:{name}, bits: 4, group_size: 64", module.weight.shape)
			# 			module.weight = torch.nn.Parameter(qdq_group(module.weight,4, 64,True))
			# 			flag = True
			# 			continue
			# 	if not flag:
			# 		print(f"Quantizing the dp weight of the Layer:{name}, bits: 3, group_size: 64", module.weight.shape)
			# 		module.weight = torch.nn.Parameter(qdq_group(module.weight,3, 64, True))
			else:
				flag = False
				patterns_2 = [f".{i}." for i in [22,23,24,25,25,29]]
				for pattern in patterns_2:
					if pattern in name:
						print(f"Quantizing the dp weight of the Layer:{name}, bits: 2, group_size: 64", module.weight.shape)
						module.weight = torch.nn.Parameter(qdq_group(module.weight,2, 64,True))
						flag = True
						continue
				if not flag:
					print(f"Quantizing the dp weight of the Layer:{name}, bits: 3, group_size: 64", module.weight.shape)
					module.weight = torch.nn.Parameter(qdq_group(module.weight,3, 64, True))
		# elif isinstance(module, nn.Linear):
		# 	patterns_2 = [f".{i}." for i in range(22,28)]
		# 	for pattern in patterns_2:
		# 		if pattern in name:
		# 			print(f"Quantizing the weight of the Layer:{name}, bits: 2, group_size: 64")
		# 			module.weight = torch.nn.Parameter(qdq(module.weight,2, 64))
		# 			continue
			
	return model
	
def build_model_and_tokenizer(model_name):
	kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
	tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
	return model, tokenizer

def main():
	# Declare Parser and its arguments
	'''
	python3 /auto/regrt/sw/dgundimeda/dynamic_quant_experiments/collect_stats.py --model /auto/regrt/sw/saketh/model_epoch_20_hybrid_g64_all_2_22_to_27/qwen2_7b_llamafied --limit 5
 	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, help="model name (or) model path")
	parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
	parser.add_argument("--calib_dataset",type=str,default="wikitext2",
		choices=["wikitext2", "ptb", "c4", "mix","pile"],
		help="Where to extract calibration data from.",
	)
	parser.add_argument("--observer",type=str,default="min_max",
						choices = ["min_max", "moving_ave_min_max", "abs_min_max"],
						help = "select the type of observer")
	parser.add_argument("--wbits", type=int, default=8)
	parser.add_argument("--abits", type=int, default=8)
	parser.add_argument("--limit", type=int, default=-1)
	parser.add_argument("--wquant", type=bool, default=True)

	# Parse the args passed via cmd
	args = parser.parse_args()
	
	# Set same seed for numpy, torch, torch_cuda and random modules
	seed = 10
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	# Initialize the model and tokenizer objects
	model, tokenizer = build_model_and_tokenizer(args.model)

	print("Evaluate Original Model: ")
	evaluate(model, args)
	print("*"*50)
	if args.wquant:
		model = quantize_model_weights(model)

	model = replace_norm(model)
	print(model)
 
	model, removable_handles, stat_hook = add_stat_hooks(model, args.observer)
	print("*"*50)
	print("Evaluate Weight Quant Model: ")
	evaluate(model, args)
	print("*"*50)

	stat_hook = override_dtype(stat_hook)

	for handle in removable_handles:
		handle.remove()

	scale_offset = calculate_scale_offset(stat_hook, args.observer)
	with open("/auto/worka/dgundimeda/scale_offset_const_scale_32bit.json","w") as json_file:
		json.dump(scale_offset, json_file)
 
	model, removable_handles, qdq_hook = add_output_hooks(model, scale_offset)
	print("*"*50)
	print("Evaluate Act Quant Model: ")
	evaluate(model, args)
	print("*"*50)
 
	for handle in removable_handles:
		handle.remove()
 
if __name__ == "__main__":
	print(sys.argv)
	main()