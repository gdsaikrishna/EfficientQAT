import sys
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld_test/dv2/python")
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld_test/dv2/python/caffe/proto")
import caffe
import numpy as np
source_prototxt = ""
source_caffemodel = ""
target_prototxt= ""
# Load the source model
source_net = caffe.Net(source_prototxt, source_caffemodel, caffe.TEST)

# Load the target model
target_net = caffe.Net(target_prototxt, caffe.TEST)
# Copy the parameters from the source model to the target model
for param_name in source_net.params.keys():
    print(param_name)
    if param_name == "_lm_head_MatMul_ara_inrp":
        zeros_data = np.zeros((21,4096,1,1))
        target_net.params[param_name][0].data[...] = np.concatenate((source_net.params[param_name][0].data, zeros_data)).astype(np.float32)
    elif param_name == "_model_embed_tokens_Gather":
        zeros_data = np.zeros((21,4096))
        target_net.params[param_name][0].data[...] = np.concatenate((source_net.params[param_name][0].data, zeros_data)).astype(np.float32)
    elif param_name in target_net.params.keys():
        for i in range(len(source_net.params[param_name])):
            target_net.params[param_name][i].data[...] = source_net.params[param_name][i].data[...]

# Save the target model's weights to a new caffemodel file
import numpy as np
print("Before shape: ",source_net.params["_lm_head_MatMul_ara_inrp"][0].data.shape)

print("After shape: ",target_net.params["_lm_head_MatMul_ara_inrp"][0].data.shape)
target_net.save('target_model.caffemodel')
