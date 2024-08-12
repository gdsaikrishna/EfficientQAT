import sys
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld6/dv2/python")
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld6/dv2/python/caffe/proto")
import caffe

prototxt = "/auto/regrt/sw/titash/qwen_converted/params/nnconvert.prototxt"
caffemodel = "/auto/regrt/sw/titash/qwen_converted/params/nnconvert.caffemodel"
inet = caffe.Net(prototxt, caffemodel, caffe.TEST)
import numpy as np
print("Before shape: ",inet.params["_lm_head_MatMul_ara_inrp"][0].data.shape)
zeros_data = np.zeros(21,4096,1,1)
inet.params["_lm_head_MatMul_ara_inrp"][0].data = np.concatenate((inet.params["_lm_head_MatMul_ara_inrp"][0].data, zeros_data))
zeros_data = np.zeros(21,4096)
inet.params["_model_embed_tokens_Gather"][0].data = np.concatenate((inet.params["_model_embed_tokens_Gather"][0].data, zeros_data))
print("After shape: ",inet.params["_lm_head_MatMul_ara_inrp"][0].data.shape)

inet.save_hdf5('nnconvert.caffemodel')