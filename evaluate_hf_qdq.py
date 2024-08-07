import argparse
import json
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from categories import categories, subcategories

choices = ["A", "B", "C", "D"]

all_outputs = ['embed_tokens', 'input_layernorm', 'post_attention_layernorm', 'norm', 'lm_head'
			   'q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkt_proj', 'sfmx' , 'attn_output_layer', 'rotary_emb',
			   'gate_proj', 'up_proj', 'down_proj' , 'act_fn', 'eltmul',
			   'attn_residue', 'mlp_residue']
outputs_8bit = ['q_proj','k_proj','v_proj','qkt_proj','attn_output_layer','gate_proj' ,'act_fn','up_proj','norm', 'rotary_pos_emb']
outputs_16bit = ['down_proj','embed_tokens','attn_residue','mlp_residue','eltmul','sfmx','o_proj','lm_head']
patterns = outputs_8bit + outputs_16bit

def round_ste(x: torch.Tensor):
	return (x.round() - x).detach() + x

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


def quantize_model_weights(model, n_bits = 8, group_size = None):
	for name,module in model.named_modules():
		if isinstance(module, nn.Linear):
			patterns_2 = [f".{i}." for i in [22,23,24,25,26,29]]
			if "lm_head" in name:
				if 'weight' in [name1 for name1, param in module.named_parameters()]:
					# print(f"Quantizing the lm_head weight of the Layer:{name}, bits: 4, group_size: 64")
					module.weight = torch.nn.Parameter(qdq(module.weight,4, 64, True))
			else:
				flag = False
				patterns_2 = [f".{i}." for i in [22,23,24,25,26,27]]
				for pattern in patterns_2:
					if pattern in name:
						module.weight = torch.nn.Parameter(qdq(module.weight,2, 64, True))
						flag = True
						continue
				if not flag:
					module.weight = torch.nn.Parameter(qdq(module.weight,3, 64, True))
			
	return model


def calibrate_model(model, dataset,samples_count=146):
	# model.to(device)
	print("Model in device: ", model.device)
	seqlen = 2048
	testloader = None
	cache_testloader = '/home/liliang11/Desktop/EfficientQAT/cache/testloader_qwen2_7b_wikitext2_all.cache'
	if os.path.exists(cache_testloader): 
		testloader = torch.load(cache_testloader)
		print(f"load calibration from {cache_testloader}")
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


def format_subject(subject):
	l = subject.split("_")
	s = ""
	for entry in l:
		s += " " + entry
	return s


def format_example(df, idx, include_answer=True):
	prompt = df.iloc[idx, 0]
	k = df.shape[1] - 2
	for j in range(k):
		prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
	prompt += "\nAnswer:"
	if include_answer:
		prompt += " {}\n\n".format(df.iloc[idx, k + 1])
	return prompt


def gen_prompt(train_df, subject, k=-1):
	prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
		format_subject(subject)
	)
	if k == -1:
		k = train_df.shape[0]
	for i in range(k):
		prompt += format_example(train_df, i)
	return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
	cors = []
	all_probs = []
	answers = choices[: test_df.shape[1] - 2]

	for i in range(test_df.shape[0]):
		# get prompt and make sure it fits
		k = args.ntrain
		prompt_end = format_example(test_df, i, include_answer=False)
		train_prompt = gen_prompt(dev_df, subject, k)
		prompt = train_prompt + prompt_end

		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

		while input_ids.shape[-1] > 2048:
			k -= 1
			train_prompt = gen_prompt(dev_df, subject, k)
			prompt = train_prompt + prompt_end
			input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
				model.device
			)

		label = test_df.iloc[i, test_df.shape[1] - 1]

		logits = model(input_ids=input_ids).logits[0, -1]

		probs = (
			torch.nn.functional.softmax(
				torch.tensor(
					[
						logits[tokenizer("A").input_ids[-1]],
						logits[tokenizer("B").input_ids[-1]],
						logits[tokenizer("C").input_ids[-1]],
						logits[tokenizer("D").input_ids[-1]],
					]
				).float(),
				dim=0,
			)
			.detach()
			.cpu()
			.numpy()
		)
		pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

		cor = pred == label
		cors.append(cor)
		all_probs.append(probs)

	acc = np.mean(cors)
	cors = np.array(cors)

	all_probs = np.array(all_probs)
	print("Average accuracy {:.3f} - {}".format(acc, subject))

	return cors, acc, all_probs


def main(args):
	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		torch_dtype=torch.float16,
		load_in_8bit=False,
		low_cpu_mem_usage=True,
		device_map="auto",
	)
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	output_stats = dump_samples_stats(model)
	dump_json("output_stats.json",output_stats)
	output_scale_offsets = calculate_scale_offsets(output_stats)
	with open("output_scale_offsets.json", 'w') as json_file:
		json.dump(output_scale_offsets, json_file)
	if args.quantize_weights:
		model = quantize_model_weights(model, 4, 64)
		print("Post Quantizing weights: ")
		calibrate_model(model, args.calib_dataset,10)

	if args.quantize_activations:
		model ,qdq_hooks = add_qdq_hooks(model, None, output_scale_offsets)
	model.eval()
	subjects = sorted(
		[
			f.split("_test.csv")[0]
			for f in os.listdir(os.path.join(args.data_dir, "test"))
			if "_test.csv" in f
		]
	)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
		os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

	all_cors = []
	subcat_cors = {
		subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
	}
	cat_cors = {cat: [] for cat in categories}

	for subject in subjects:
		dev_df = pd.read_csv(
			os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
		)[: args.ntrain]
		test_df = pd.read_csv(
			os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
		)

		cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
		subcats = subcategories[subject]
		for subcat in subcats:
			subcat_cors[subcat].append(cors)
			for key in categories.keys():
				if subcat in categories[key]:
					cat_cors[key].append(cors)
		all_cors.append(cors)

		test_df["{}_correct".format(args.model)] = cors
		for j in range(probs.shape[1]):
			choice = choices[j]
			test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
		test_df.to_csv(
			os.path.join(
				args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
			),
			index=None,
		)

	results = {"subcategories": {}, "categories": {}}
	for subcat in subcat_cors:
		subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
		results["subcategories"][subcat] = subcat_acc
		print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

	for cat in cat_cors:
		cat_acc = np.mean(np.concatenate(cat_cors[cat]))
		results["categories"][cat] = cat_acc
		print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
	weighted_acc = np.mean(np.concatenate(all_cors))
	results["weighted_accuracy"] = weighted_acc
	print("Average accuracy: {:.3f}".format(weighted_acc))

	results_file = os.path.join(
		args.save_dir, "accuracies_{}.json".format(args.model.replace("/", "_"))
	)
	with open(results_file, "w") as f:
		json.dump(results, f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ntrain", "-k", type=int, default=5)
	parser.add_argument("--data_dir", "-d", type=str, default="data")
	parser.add_argument("--save_dir", "-s", type=str, default="results")
	parser.add_argument("--model", "-m", type=str)
	parser.add_argument("--quantize_weights",action = "store_true")
	parser.add_argument("--quantize_activations",action = "store_true")
	args = parser.parse_args()
	main(args)