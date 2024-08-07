# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

import os
import re
from tqdm import tqdm
import random, functools
import numpy as np
import torch, json
from transformers import LlamaForCausalLM, LlamaTokenizer
from evaluator import Evaluator
import torch.nn as nn

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
	calibrate_model(model, "wikitext2",10)
	
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

class Llama_Evaluator(Evaluator):
	def __init__(self, choices, k, model_path, device, temperature=0.2):
		super(Llama_Evaluator, self).__init__(choices, model_path, k)
		load_type = torch.float16
		self.model_path = model_path
		self.device = device
		self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
		self.model = LlamaForCausalLM.from_pretrained(
			model_path,
			load_in_8bit=False,
			torch_dtype=load_type,
			low_cpu_mem_usage=True,
			device_map='auto')
		
		output_scale_offsets = read_json("")
		quantize_weights = True
		if quantize_weights:
			self.model = quantize_model_weights(self.model, 4, 64)
			print("Post Quantizing weights: ")
			calibrate_model(self.model, "wikitext2",10)

		quantize_activations = True
		if quantize_activations:
			self.model ,qdq_hooks = add_qdq_hooks(self.model, None, output_scale_offsets)
		self.generation_config = dict(
			temperature=temperature,
			top_k=40,
			top_p=0.9,
			do_sample=True,
			num_beams=1,
			repetition_penalty=1.1,
			max_new_tokens=20
		)

		self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
		self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
		self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
		self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
		self.A_id = self.tokenizer.encode("：A")[-1]
		self.B_id = self.tokenizer.encode("：B")[-1]
		self.C_id = self.tokenizer.encode("：C")[-1]
		self.D_id = self.tokenizer.encode("：D")[-1]


	def eval_subject(self, subject_name,
			test_df,
			dev_df=None,
			few_shot=False,
			cot=False,
			save_result_dir=None,
			with_prompt=False,
			constrained_decoding=False,
			do_test=False):
		all_answers = {}
		if constrained_decoding is True:
			self.generation_config['output_scores'] = True
			self.generation_config['return_dict_in_generate'] = True
			self.generation_config['max_new_tokens'] = 1
			self.generation_config['top_p'] = 1.0
			self.generation_config['top_k'] = 0

		correct_num = 0
		if save_result_dir:
			result = []
			score = []
		if few_shot:
			history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
		else:
			history = ''
		answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
		for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
			question = self.format_example(row, include_answer=False, cot=cot,with_prompt=with_prompt)
			instruction = history + question
			if with_prompt:
				prompt_template = (
					"Below is an instruction that describes a task. "
					"Write a response that appropriately completes the request.\n\n"
					"### Instruction:\n{instruction}\n\n### Response: ")

				instruction = prompt_template.format_map({'instruction': instruction,'subject':subject_name})

			inputs = self.tokenizer(instruction, return_tensors="pt")
			generation_output = self.model.generate(
					input_ids = inputs["input_ids"].to(self.device),
					attention_mask = inputs['attention_mask'].to(self.device),
					eos_token_id=self.tokenizer.eos_token_id,
					pad_token_id=self.tokenizer.pad_token_id,
					**self.generation_config
				)

			batch_size, length = inputs.input_ids.shape
			if constrained_decoding is True:
				logits = generation_output.scores[0][0]

				logits = logits.float().cpu().detach()
				choices1_logits = logits[[self.sA_id,self.sB_id,self.sC_id,self.sD_id]]
				choices2_logits = logits[[self.A_id,self.B_id,self.C_id,self.D_id]]
				choicesAll_logits = (choices1_logits + choices2_logits).numpy()
				assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
				ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
				response = self.tokenizer.decode([logits.argmax(-1).item()])
			else:
				response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
				ans, direct_extract = self.extract_answer(row, response)
			if ans == answers[row_index]:
				correct_num += 1
				correct = 1
			else:
				correct = 0
			print(f"\n=======begin {str(row_index)}=======")
			print("question: ", question)
			print("response: ", response)
			print("ans: ", ans)
			print("ground truth: ", answers[row_index], "\n")
			if save_result_dir:
				result.append(response)
				score.append(correct)
			print(f"=======end {str(row_index)}=======")

			all_answers[str(row_index)] = ans

		correct_ratio = 100*correct_num/len(answers)

		if save_result_dir:
			test_df['model_output'] = result
			test_df['correctness'] = score
			test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

		return correct_ratio, all_answers

	def format_example(self, line, include_answer=True, cot=False, with_prompt=False):
		example = line['question']
		for choice in self.choices:
			example += f'\n{choice}. {line[f"{choice}"]}'
		if include_answer:
			if cot:
				example += "\n答案：让我们一步一步思考，\n" + \
					line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
			else:
				example += '\n答案：' + line["answer"] + '\n\n'
		else:
			if with_prompt is False:
				if cot:
					example += "\n答案：让我们一步一步思考，\n1."
				else:
					example += '\n答案：'
			else:
				if cot:
					example += "\n答案是什么？让我们一步一步思考，\n1."
				else:
					example += '\n答案是什么？ '
		return example

	def generate_few_shot_prompt(self, subject, dev_df, cot=False):
		prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
		k = self.k
		if self.k == -1:
			k = dev_df.shape[0]
		for i in range(k):
			prompt += self.format_example(
				dev_df.iloc[i, :],
				include_answer=True,
				cot=cot
			)
		return prompt

	def extract_answer(self, line, gen_ans):
		m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
		if len(m) > 0 and m[-1] in self.choices:
			return m[-1], True
		answer_patterns = [
			r'([ABCD])是正确的',
			r'选项([ABCD])正确',
			r'答案为([ABCD])',
			r'答案是([ABCD])',
			r'答案([ABCD])',
			r'选择([ABCD])',
			r'答案：([ABCD])',
			r'选择答案([ABCD])'
		]
		# RE extraction
		for answer_pattern in answer_patterns:
			m = re.search(answer_pattern, gen_ans, re.M)
			if m:
				answer = m.group(1)
				return answer, False
		# only containing one choice-character
		m = re.findall(r'[ABCD]', gen_ans, re.M)
		if len(m) >= 1:
			answer = m[0]
			return answer, False
		# only containing one choice-context
		choices_dict = {}
		pattern = ""
		for c in self.choices:
			choices_dict[str(line[f'{c}'])] = c
			pattern += re.escape(str(line[f'{c}']))+"|"
		pattern = pattern[:-1]
		m = re.findall(pattern, gen_ans, re.M)
		print("w/ escape:",repr(pattern),gen_ans,(len(m)>=1))
		if len(m) >= 1:
			answer = choices_dict[m[0]]
			return answer, False
		return  random.choice('ABCD'), False