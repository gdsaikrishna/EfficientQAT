import onnx
import numpy as np

def getModelWts():
	model_path = ""
	model = onnx.load(model_path)
	print("load done")

	target_node_name = "/model/embed_tokens/Gather"
	target_node = None
	for node in model.graph.node:
		if node.name == target_node_name:
			target_node = node
			break

	input_node_index = None
	for i, input_node_name in enumerate(target_node.input):
		if input_node_name == "model.embed_tokens.weight":
			input_node_index = i
			break
	
	wt_array = None
	if input_node_index is not None:
		input_tensor_name = model.graph.initializer[input_node_index].name
		for initializer in model.graph.initializer:
			if initializer.name == input_tensor_name:
				tensor_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
				print(tensor_data.shape)
				zeros_to_append = np.zeros(21*4096, dtype=np.float32)
				tensor_data = np.append(tensor_data, zeros_to_append)
				dims = initializer.dims
				dims[0] = dims[0]+21
				wt_array = tensor_data.reshape(dims)
				np.save("model_embed_tokens_weight.npy", wt_array)
				break
	return 

getModelWts()
