#!/bin/env python3
import os, onnx, numpy as np, copy, shutil, argparse, yaml
import sys as _sys
# _sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../.././../../../src/dv/analyzer/python')
# _sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../.././../../../src/dv/analyzer/python/caffe/proto')

_sys.path.append('/auto/worka/bhavan/git/ara2/l_t1/arasw/bld_test/dv2/python/')
_sys.path.append('/auto/worka/bhavan/git/ara2/l_t1/arasw/bld_test/dv2/python/caffe/proto')
# _sys.path.append('/auto/worka/shubham/g/s0/sw/br0/dv/python/')
# _sys.path.append('/auto/worka/shubham/g/s0/sw/br0/dv/python/caffe/proto')

import caffe
from caffe import layers as L
import caffe.proto.caffe_pb2 as caffe_pb2
from caffe import params as P
from caffe import to_proto


class Node:
    def __init__(self, name, bottoms=[], tops=[], type='', text=''):
        self.name = name
        self.bottoms = bottoms
        self.tops = tops
        self.type = type
        self.text = text
    
    def __repr__(self):
        return "name: "+self.name+'\n' +"bottoms: "+','.join(self.bottoms)+'\n' +"tops: "+','.join(self.tops)+'\n' +"type: "+self.type+'\n' +"text: "+''.join(self.text)
    
    def __str__(self):
        return self.__repr__()

class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.name_node = dict()
        self.bot_name = dict()
        self.top_name = dict()

def node_from_lines(lines):
    combined = ''.join(lines[1:-1])
    aa = combined.find('{')
    bb = combined.rfind('}')
    combined = combined[:aa] + combined[bb+1:]
    def get_field(lines, name):
        res = []
        pos = 0
        while lines[pos:].find(name+':') != -1:
            aa = lines[pos:].find(name+':')
            bb = lines[pos + aa:].find('\n')
            cc = lines[pos + aa + len(name) + 1 :pos + aa + bb]
            res.append(cc[cc.find('"')+1:cc.rfind('"')])
            pos += aa + bb + 1
        return res
    name = get_field(combined, 'name')[0]
    bottoms = get_field(combined, 'bottom')
    tops = get_field(combined, 'top')
    type = get_field(combined, 'type')[0]
    return Node(name, bottoms, tops, type, lines)

def parse_prototxt(path):
    gr = Graph()
    with open(path,'r') as f:
        lines = f.readlines()
    # all([i.endswith('\n') for i in lines])
    # all([i.count('\n')==1 for i in lines])
    ln = 0
    while(ln < len(lines)):
        if lines[ln] == "layer {\n":
            count = 1
            ln_end = ln
            while count != 0:
                ln_end += 1
                count += lines[ln_end].count('{') - lines[ln_end].count('}')
            node = node_from_lines(lines[ln:ln_end+1])
            gr.nodes.append(node)
            if node.type == 'Input':
                gr.inputs.append(node.name)
            ln = ln_end + 1
        else:
            ln += 1
    return gr

def fill_dicts(graph):
    for n in graph.nodes:
        graph.name_node[n.name] = n
        for b in n.bottoms:
            if b not in graph.bot_name:
                graph.bot_name[b] = [n.name]
            else:
                graph.bot_name[b].append(n.name)
        for b in n.tops:
            if b not in graph.top_name:
                graph.top_name[b] = [n.name]

# def ltr(graph, name, res, mode):
#     if name not in graph.name_node:
#         return
#     node = graph.name_node[name]
#     if name in res:
#         return
#     if mode == 1:
#         for t in node.tops:
#             for c in graph.bot_name.get(t,[]):
#                 ltr(graph, c, res, mode)
#     elif mode == -1:
#         for t in node.bottoms:
#             for c in graph.top_name.get(t,[]):
#                 ltr(graph, c, res, mode)
#     res.add(name)

# def get_names(graph, inputnn, mode = 0):
#     res = set()
#     for n in inputnn:
#         ltr(graph, n, res, mode)
#     return res


def get_names(graph, inputnn, mode = 0):
    res = set()
    stk = []
    for n in inputnn:
        if n in graph.name_node:
            stk.append(graph.name_node[n])
    while len(stk):
        cur = stk[-1]
        rels = []
        if mode == 1:
            for t in cur.tops:
                for i in graph.bot_name.get(t,[]):
                    if (i in graph.name_node) and (i not in res):
                        rels.append(graph.name_node[i])
                        stk.append(graph.name_node[i])
        if mode == -1:
            for t in cur.bottoms:
                for i in graph.top_name.get(t,[]):
                    if (i in graph.name_node) and (i not in res):
                        rels.append(graph.name_node[i])
                        stk.append(graph.name_node[i])
        if len(rels) == 0:
            res.add(cur.name)
            stk.pop()
    return res

def get_shapes(prototxt, caffemodel, graph):
    inet = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for i in graph.inputs:
        blob_name = graph.name_node[i].tops[0]
        blob_shape = inet.blobs[blob_name].data.shape
        inet.blobs[blob_name].data[...] = np.random.randn(*blob_shape)
    # inet.forward()
    shape_dict = {}
    for i in inet.blobs:
        shape_dict[i] = inet.blobs[i].data.shape
    return shape_dict, inet

def get_new_graph(graph, subg_names):
    new_graph = Graph()
    lt = [n for n in graph.nodes]
    for n in lt:
        if n.name in subg_names:
            new_graph.nodes.append(n)
    fill_dicts(new_graph)
    return new_graph

def replace_text(inp_text, inp_text_raw, field, data):
    pos = []
    for i, j in enumerate(inp_text_raw):
        if field+':' in j:
            p1 = j.find(field+':')
            if p1 == 0 or j[p1-1] == ' ':
                pos.append(i)
    inp_text = inp_text[:pos[0]] + inp_text[pos[-1]+1:]
    txt_l = inp_text_raw[pos[0]].split(field+':')[0]
    for i in reversed(data):
        if type(i) == type('ab'):
            txt_f = '"' + i + '"'
        else:
            txt_f = str(i)
        inp_text.insert(pos[0], txt_l + field+': ' + txt_f + '\n')
    return inp_text

def fill_new_inputs(new_proto_txt, blob_wo_inputs, shape_dict, graph, res_a):
    inp_text_raw = graph.name_node[graph.inputs[0]].text
    res_b = []
    for i in blob_wo_inputs:
        inp_text = [str(i) for i in graph.name_node[graph.inputs[0]].text]
        inp_text = replace_text(inp_text, inp_text_raw, 'dim', shape_dict[i[0]])
        inp_text = replace_text(inp_text, inp_text_raw, 'name', i[1])
        inp_text = replace_text(inp_text, inp_text_raw, 'top', [i[0]])
        new_proto_txt = new_proto_txt + inp_text
        res_b.append([i[1][0], shape_dict[i[0]]])
    res_a.append(res_b)
    return new_proto_txt

def fill_other_nodes(new_proto_txt, subg_names, graph):
    for i in graph.nodes:
        if i.name in subg_names:
            new_proto_txt = new_proto_txt + graph.name_node[i.name].text
    return new_proto_txt

def save_inputs(blob_wo_inputs, inet, res_a, dst_path):
    res_a.append(dst_path+'_imgs')
    shutil.rmtree(dst_path+'_imgs', ignore_errors=True)
    os.makedirs(dst_path+'_imgs/1', exist_ok=True)
    for i in blob_wo_inputs:
        blob = inet.blobs[i[1][0]].data
        blob.tofile(dst_path+'_imgs/1/'+i[1][0]+'.bin')

def abc(graph, shape_dict, inet, inputnn, outputnn, dst_path, res_a):
    child_names = get_names(graph, inputnn, 1)
    par_names = get_names(graph, outputnn, -1)
    subg_names = child_names.intersection(par_names)
    graph_new = get_new_graph(graph, subg_names)
    blob_wo_inputs = []
    for i in graph_new.bot_name:
        if i not in graph_new.top_name:
            blob_wo_inputs.append([i, graph.top_name[i]])
    new_proto_txt = []
    new_proto_txt = fill_new_inputs(new_proto_txt, blob_wo_inputs, shape_dict, graph, res_a)
    # save_inputs(blob_wo_inputs, inet, res_a, dst_path)
    new_proto_txt = fill_other_nodes(new_proto_txt, subg_names, graph)
    with open(dst_path+'.prototxt', 'w') as f:
        f.writelines(new_proto_txt)
    onet = caffe.Net(dst_path+'.prototxt', caffe.TEST)
    for l in onet.params.keys():
        for i in range(len(onet.params[l])):
            onet.params[l][i].data[:] = inet.params[l][i].data[:]
    onet.save_hdf5(dst_path + '.caffemodel')


# src_path = '/auto/worka/shubham/files/small1/24_5/qwen/models/llama2_7b_batch1_accurate_no_embeddings_c2c_1/nnconvert.prototxt'
# src_path = '/auto/worka/shubham/g/s3/arasw/sw1/candid/test/networks/dv_cfg0/caffe/modelzoo/llama2_split1/__run.default/nnconvert.default/params/nnconvert.prototxt'
src_path = '/auto/worka/shubham/files/small1/24_5/qwen/models/qwen_llm_qat_1024/nnconvert.prototxt'

# dst_path = 'out'

def get_sections(graph, inputnn, outputnn):
    for n in graph.nodes:
        if n.type == 'Ara2LayerNorm':
            if 'post_attention' not in n.name:
                inputnn.append(n.name)
            else:
                rsh_node = graph.name_node[graph.top_name[n.bottoms[0]][0]]
                outputnn.append([i for i in graph.bot_name[rsh_node.bottoms[0]] if i != rsh_node.name][0])
    if len(inputnn) > len(outputnn): inputnn.pop()
    last = outputnn[-1]
    succs = graph.bot_name.get(graph.name_node[last].tops[0], [])
    inputnn.append(succs[0])
    while len(succs):
        last = succs[0]
        succs = graph.bot_name.get(graph.name_node[last].tops[0], [])
    outputnn.append(last)
    

def split(network_file='', model_file='', dst_path='', idx_min=0, idx_max=0):
    graph = parse_prototxt(network_file)
    fill_dicts(graph)
    shape_dict, inet = get_shapes(network_file, model_file, graph)
    inputnn, outputnn = [], []
    get_sections(graph, inputnn, outputnn)
    # inputnn = ['_model_layers_0_input_layernorm_Mul_1_ara_reshape_0', '_model_layers_1_input_layernorm_Mul_1_ara_reshape_0', '_model_layers_2_input_layernorm_Mul_1_ara_reshape_0']
    # outputnn = ['_model_layers_0_Add_1', '_model_layers_1_Add_1','_model_layers_2_Add_1']
    res = []
    # for i in [0,1,2,32]:
    for i in range(idx_min, min(idx_max+1, len(inputnn))):
        res_a = [i, [outputnn[i]], dst_path+str(i)]
        abc(graph, shape_dict, inet, [inputnn[i]], [outputnn[i]], dst_path+str(i), res_a)
        res.append(res_a)
    return res
    

def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, dict) and isinstance(dict1[key], dict):
                deep_merge(dict1[key], value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1

def mutate_llama2_no_embeddings_splits(output_dir, config={}):
    import shutil
    from jinja2 import Template
    import random
    out_path = output_dir
    #os.makedirs(out_path, exist_ok=True)
    # split = importlib.import_module('dv.sw.ops.llama2_no_embeddings_splits.split')

    # os.makedirs(mutate_path, exist_ok=True)
    res = split(config['network_file'], config['model_file'], out_path+'/split_', config['decoder_start'], config['decoder_end'])
    offset_dict = {}
    def fill_offset_dict(value, entry):
        if type(entry) == type(0):
            if entry != -1: 
                offset_dict[entry] = value
            return
        entry = [i for i in entry.split(',')]
        for i in entry:
            if '-' not in i:
                offset_dict[int(i)] = value
            else:
                lw, hg = int(i.split('-')[0]), int(i.split('-')[1])
                for j in range(lw, hg+1):
                    offset_dict[j] = value
    fill_offset_dict(4, config['scale_offset_4bit'])
    fill_offset_dict(3, config['scale_offset_3bit'])
    fill_offset_dict(2, config['scale_offset_2bit'])

    # with open("/dv2/templates/qwen_prmpt_prc_tmpl.j2") as file_:
    #     prmpt_prc_template = Template(file_.read())
    # with open("/dv2/templates/qwen_tkn_gen_tmpl.j2") as file_:
    #     tkn_gen_template = Template(file_.read())
    # with open("/dv2/templates/qwen_prmpt_lm_head_tmpl.j2") as file_:
    #     prompt_lm_head_template = Template(file_.read())
    # with open("/dv2/templates/qwen_token_lm_head_tmpl.j2") as file_:
    #     token_lm_head_template = Template(file_.read())
    # with open("/dv2/templates/scale_offset_prmpt_prc.j2") as file_:
    #     scale_template_1 = Template(file_.read())
    # with open("/dv2/templates/scale_offset_tkn_gen.j2") as file_:
    #     scale_template_2 = Template(file_.read())

    with open("templates/qwen_prmpt_prc_tmpl.j2") as file_:
        prmpt_prc_template = Template(file_.read())
    with open("templates/qwen_tkn_gen_tmpl.j2") as file_:
        tkn_gen_template = Template(file_.read())
    with open("templates/qwen_prmpt_lm_head_tmpl.j2") as file_:
        prompt_lm_head_template = Template(file_.read())
    with open("templates/qwen_token_lm_head_tmpl.j2") as file_:
        token_lm_head_template = Template(file_.read())
    with open("templates/scale_offset_prmpt_prc.j2") as file_:
        scale_template_1 = Template(file_.read())
    with open("templates/scale_offset_tkn_gen.j2") as file_:
        scale_template_2 = Template(file_.read())

    prompt_offset_file = ''
    if 'prompt_scale_offset_adjust_file' in config:
        prompt_offset_file = config['prompt_scale_offset_adjust_file']
    token_offset_file = ''
    if 'token_scale_offset_adjust_file' in config:
        token_offset_file = config['token_scale_offset_adjust_file']
    prompt_imgs_q = config['prompt_image_quant_path'] if 'prompt_image_quant_path' in config else ''
    prompt_imgs_v = config['prompt_image_verif_path'] if 'prompt_image_verif_path' in config else ''
    token_imgs_q = config['token_image_quant_path'].split(' ') if 'token_image_quant_path' in config else ''
    token_imgs_v = config['token_image_verif_path'].split(' ') if 'token_image_verif_path' in config else ''
    
    all_splits_p = {}
    all_splits_t = {}
    last_i = -1
    for k in res:
        i, j = k[0], k[1:]
        for z in range(2):
            prefix = ['prmpt_prc_', 'tkn_gen_'][z]
            all_splits_x = [all_splits_p, all_splits_t][z]
            offset_file = [prompt_offset_file, token_offset_file][z]
            scale_template = [scale_template_1, scale_template_2][z]
            out_blob_name = ['_residue_rshp_3', '_Add_1'][z]
            for tk in range(config[prefix+'inp_min'], config[prefix+'inp_max']+1, config[prefix+'inp_stp']):
                suff = prefix + str(tk) + '_split_' + str(i)
                curr_path = os.path.join(out_path, prefix+str(tk)+'/split_'+str(i))
                if tk not in all_splits_x: 
                    all_splits_x[tk] = []
                all_splits_x[tk].append([i, curr_path, suff])
                os.makedirs(curr_path, exist_ok=True)
                # gen files from templates
                if z == 0:
                    if i!=32:
                        content = prmpt_prc_template.render(idx = str(i), tkn = str(tk))
                    else:
                        content = prompt_lm_head_template.render(idx = str(i), tkn = str(tk))
                elif z == 1:
                    if i!=32:
                        content = tkn_gen_template.render(idx = str(i), cntx = str(tk))
                    else:
                        content = token_lm_head_template.render(idx = str(i), cntx = str(tk))
                with open(curr_path+'/'+suff+'.prototxt', mode="w", encoding="utf-8") as message:
                    message.write(content)
                # saving scale offset file
                scale_offset_path = curr_path+'/'+suff+'_scale_offset.yaml'
                if offset_file == '':
                    content = scale_template.render(val = str(offset_dict[i]), idx = str(i))
                    with open(scale_offset_path, mode="w", encoding="utf-8") as message:
                        message.write(content)
                else:
                    # scale_offset_data = dv.sw.utils.utils.read_yaml_file(offset_file)
                    with open(offset_file, 'r') as file:
                        scale_offset_data = yaml.safe_load(file)
                    if last_i != -1:
                        lyr_name = '_model_layers_'+str(last_i)+out_blob_name
                        scale_offset_data['LayerNames']['input_ids_embedded']['offset'] = scale_offset_data['LayerNames'][lyr_name]['offset']
                        scale_offset_data['LayerNames']['input_ids_embedded']['scale'] = scale_offset_data['LayerNames'][lyr_name]['scale']
                    # dv.sw.utils.utils.write_yaml_file(scale_offset_path, scale_offset_data)
                    with open(scale_offset_path, 'w') as file:
                        yaml.dump(scale_offset_data, file, default_flow_style=False)
                # gen config file
                shutil.copy(j[1]+'.caffemodel', curr_path+'/'+suff+'.caffemodel')
                if z == 0:
                    inputs_all = [['input_ids_embedded', (1, 1, tk, 4096)], ['valid_till', (1, 1, 1, 1)], ['position_ids_embedded', (1, 1, 256, tk)]]
                elif z == 1:
                    inputs_all = [['input_ids_embedded', (1, 1, 1, 4096)], ['valid_till', (1, 1, 1, 1)], ['position_ids_embedded', (1, 1, 1, 256)], ['past_values_'+str(i), (1, 1, tk, 4096)], ['past_keys_'+str(i), (1, 1, tk, 4096)]]
                if i==32:
                    inputs_all = inputs_all[:1]

                config_yaml = {}
                config_yaml["network"] = {
                    "src_model_type": "caffe",
                    "inet": curr_path+'/'+suff+'.prototxt',
                    "iwt": curr_path+'/'+suff+'.caffemodel',
                    "dim": ':'.join([','.join([str(h) for h in k[1]]) for k in inputs_all]),
                    "images": {
                        "quantize": curr_path+'/imgs_q',
                        "verify": curr_path+'/imgs_v',
                        "tags": "/auto/share/sw/common/data/classification/imagenet/groundtruth/tags.txt",
                    },
                }
                config_yaml['dvconvert'] = {
                    "inode": ':'.join([k[0] for k in inputs_all]),
                }

                config_yaml['dvnc'] = {
                        "scale_offset_adjust_file": scale_offset_path,
                        "qmode": 9,
                        "input_type" : "valid_till:uint16"
                }
                config_yaml['out'] = curr_path

                with open('sample.yaml', 'r') as sfile:
                    template_yaml_dict = yaml.safe_load(sfile)
                merged_dict = deep_merge(template_yaml_dict.copy(), config_yaml)
                # template_yaml_dict = {**template_yaml_dict, **config_yaml}
                # dv.sw.utils.utils.write_yaml_file(curr_path+'/config_ara2.yaml', config_yaml)
                with open(curr_path+'/dvrun_config.yaml', 'w') as file:
                    yaml.dump(merged_dict, file, default_flow_style=False)
                # s = os.path.join(dv.sw.utils.sys.rundir(), mutate_file)
                # d = os.path.join(curr_path, mutate_file)
                # dv.sw.utils.sys.symlink(s, d)
        last_i = i
        os.remove(j[1]+'.prototxt')
        os.remove(j[1]+'.caffemodel')
    # gen input images
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../src/dv/analyzer/python')
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../.././../src/dv/analyzer/python/caffe/proto')
    import caffe
    for z in range(2):
        all_splits_x = [all_splits_p, all_splits_t][z]
        out_blob_name = ['_residue_rshp_3', '_Add_1'][z]
        out_analyzer_blob_name = [['_residue_rshp_3-45', '_residue_rshp_3-60'],['_Add_1-42', '_Add_1-53']][z]
        cust_image_arg_name_q = ['prompt_images_quant_path', 'token_image_quant_path'][z]
        cust_image_arg_name_v = ['prompt_images_verif_path', 'token_image_verif_path'][z]
        offset_file = [prompt_offset_file, token_offset_file][z]
        for tk in all_splits_x:
            last_i = all_splits_x[tk][0][0]
            start_i =  all_splits_x[tk][0][0]
            inp_imgs_q, inp_imgs_v = {}, {}
            for i in range(len(all_splits_x[tk])):
                j, curr_path, suff = all_splits_x[tk][i]
                # gen inp images for for inputs
                if z == 0:
                    inputs_all = [['input_ids_embedded', (1, 1, tk, 4096)], ['valid_till', (1, 1, 1, 1)], ['position_ids_embedded', (1, 1, 256, tk)]]
                elif z == 1:
                    inputs_all = [['input_ids_embedded', (1, 1, 1, 4096)], ['valid_till', (1, 1, 1, 1)], ['position_ids_embedded', (1, 1, 1, 256)], ['past_values_'+str(j), (1, 1, tk, 4096)], ['past_keys_'+str(j), (1, 1, tk, 4096)]]
                os.makedirs(curr_path+'/imgs_q/chksum0', exist_ok=True)
                os.makedirs(curr_path+'/imgs_v/chksum0', exist_ok=True)
                net = caffe.Net(curr_path+'/'+suff+'.prototxt',curr_path+'/'+suff+'.caffemodel', caffe.TEST)
                # save input iamges
                for k in inputs_all:
                    if k[0] not in net.blobs:
                        continue
                    if k[0] not in inp_imgs_v:
                        if cust_image_arg_name_v in config:
                            img_path = config[cust_image_arg_name_v]+'/'+k[0]+'.bin'
                            if not os.path.exists(img_path):
                                print(f"{img_path} doesn't exist")
                            blb = np.fromfile(img_path,dtype=np.float32).reshape(k[1])
                        else:
                            blb = np.random.randn(*k[1])
                        inp_imgs_v[k[0]] = blb
                    blb = inp_imgs_v[k[0]]
                    blb.astype('float32').tofile(curr_path+'/imgs_v/chksum0/'+k[0]+'.bin')
                    if k[0] not in inp_imgs_q:
                        if cust_image_arg_name_q in config:
                            img_path = config[cust_image_arg_name_q]+'/'+k[0]+'.bin'
                            if not os.path.exists(img_path):
                                print(f"{img_path} doesn't exist")
                            blb = np.fromfile(img_path,dtype=np.float32).reshape(k[1])
                        else:
                            blb = np.random.randn(*k[1])
                        inp_imgs_q[k[0]] = blb
                    blb = inp_imgs_q[k[0]]
                    blb.astype('float32').tofile(curr_path+'/imgs_q/chksum0/'+k[0]+'.bin')
                    net.blobs[k[0]].data[...] = blb
                # copy from previous if available
                if j > start_i:
                    prv_name = all_splits_x[tk][last_i][1]+'/imgs_v/chksum0/_model_layers_'+str(last_i)+out_blob_name+'.bin'
                    curr_name = curr_path+'/imgs_v/chksum0/'+'input_ids_embedded'+'.bin'
                    shutil.move(prv_name, curr_name)
                    prv_name = all_splits_x[tk][last_i][1]+'/imgs_q/chksum0/_model_layers_'+str(last_i)+out_blob_name+'.bin'
                    curr_name = curr_path+'/imgs_q/chksum0/'+'input_ids_embedded'+'.bin'
                    shutil.move(prv_name, curr_name)
                    blb = np.fromfile(curr_name, dtype=np.float32)
                    net.blobs['input_ids_embedded'].data[...] = np.reshape(blb,inputs_all[0][1])
                last_i = j
                # run network
                net.forward()
                # save output
                if j != all_splits_x[tk][-1][0]:
                    op_name = '_model_layers_'+str(j)+out_blob_name
                    blb = net.blobs[op_name].data
                    blb.astype('float32').tofile(curr_path+'/imgs_q/chksum0/'+op_name+'.bin')
                    # if run_analyzer == 0:
                    blb.astype('float32').tofile(curr_path+'/imgs_v/chksum0/'+op_name+'.bin')
                    # elif run_analyzer == 1:
                    #     prv_name = curr_path+'/tmp/outputs/chksum0/dequantized_fp32/_model_layers_'+str(j)+out_analyzer_blob_name[0]+'-_model_layers_'+str(j)+out_analyzer_blob_name[1]
                    #     shutil.move(prv_name, curr_path+'/imgs_v/chksum0/'+op_name+'.bin')
                    #     shutil.rmtree(curr_path+'/tmp')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to configuration file')
    parser.add_argument('--output_dir', required=True, help='path to output dir')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    mutate_llama2_no_embeddings_splits(args.output_dir, config)


if __name__ == "__main__":
    main()
