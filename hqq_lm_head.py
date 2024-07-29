import torch
from hqq.core.quantize import *
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
import warnings
from transformers import AutoTokenizer , AutoModelForCausalLM

warnings.filterwarnings("ignore")

import argparse, random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--calib_dataset",type=str,default="wikitext2", 
					choices=["wikitext2", "ptb", "c4", "mix","pile"], help="Where to extract calibration data from.")

args = parser.parse_args()

# Load the Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",torch_dtype = "auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
print("Loaded model into device:", model.device)

dummy_lm_head = torch.nn.Linear(1, 1, bias=False)
dummy_lm_head.weight.data = model.lm_head.weight
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
hqq_layer  = HQQLinear(dummy_lm_head, quant_config=quant_config, compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
breakpoint()
# def embed_hqq(self, x):
#     return torch.nn.Linear(x , padding_idx=self.padding_idx)
# model.model.lm_head.forward = lambda x: embed_hqq(model.lm_head,  x)

# Cleanup 
torch.cuda.empty_cache()
gc.collect()