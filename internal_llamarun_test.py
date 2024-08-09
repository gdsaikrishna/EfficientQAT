#!/usr/bin/env python3
# Copyright (c) 2018-23, Kinara, Inc. All rights reserved.

import argparse, os, yaml, sys, logging, ast, re, traceback, yaml, glob, multiprocessing, datetime, subprocess
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class dvrun():

  def __init__(self, config = {}, qwen_inet = None, bld_dir = ""):
    self.run_dir = ""
    self.rel_tag = "__REL__"
    self.bld_dir = bld_dir
    self.dvnc_dir = f'{self.bld_dir}/dv2/bin/'
    self.dvsim_dir = f'{self.bld_dir}/dv2/bin/'
    self.llm_mode_gen_dir = f'{self.bld_dir}/dv2/bin/'
    self.conf_dir = f'{self.bld_dir}/dv2/conf'
    self.dvconvert_dir = f'{self.bld_dir}/dv2/bin/'
    self.accuracy_dir = '/auto/regrt/sw/titash/llama2/arasw/src/dv/sdk/utils/accuracy'
    self.split_cmd_dir = "/auto/regrt/sw/titash/llama2/arasw/src/dv/sdk/utils/llama"
    self.scale_offset_cmd_dir = "/auto/regrt/sw/titash/llama2/arasw/src/dv/sdk/utils/llama"
    self.sin_cos_cmd_dir = "/auto/regrt/sw/titash/llama2/arasw/src/dv/sdk/utils/llama"
    self.gatherwts_cmd_dir = "/auto/regrt/sw/titash/llama2/arasw/src/dv/sdk/utils/llama"
    # self.validate_dir = '/dv/src/flows'
    self.docker_img = "gitea.kinara.ai/eng/sw:llamatest4-815a2db"
    # self.dvnc_cmd = "./dvnc"
    self.quantizer_cmd = "./analyzer"
    self.scheduler_cmd = "./estimator"
    self.generator_cmd = "./emitter"
    self.llm_model_gen_cmd = "./llm_model_gen"
    self.dvsim_cmd = "./dvsim"
    self.dvconvert_cmd = f"sh {self.dvconvert_dir}/nnconvert"
    self.accuracy_cmd = "python accuracy.py"
    self.validate_cmd = "python validate.py"
    self.split_cmd = "python3 split.py"
    self.scale_offset_cmd = "python scale_offset.py"
    self.sin_cos_cmd = "python sin_cos.py"
    self.gatherwts_cmd = "python getGatherWts.py"
    self.dryrun_dir = ".dryrun"
    self.dryrun_dir1 = ".dryrun1"
    self.dryrun_dir2 = ".dryrun2"
    self.converted_name = "converted"
    self.dryrun_name = "dryrun"
    self.quantizer_so_yaml = "so.yaml"
    self.quantizer_output_dir = "quantizer"
    self.scheduler_output_dir = "estimator"
    self.converter_output_dir = "converter"
    self.simulator_output_dir = "simulator"
    self.llama_output_dir = "llama"
    self.rope_output_dir = "rope_out"
    self.rope_folder_name = "rounding_rope_int8_clamped"
    self.assets_dir = "assets"
    self.params_dir = "params"
    self.stats_dir = "stats"
    self.config_yaml = "cfg0_bringup.yaml"
    self.meta_json = "super_meta.json"
    self.vpu_meta_json = "vpu_supermeta.json"
    self.power_config = "power_cfg1.json"
    self.kernel_lib = "kernel.lib"
    self.vpu_conf = "vpu"
    self.nnpsim = "nnpsim"
    self.qemubin = "qemu-system-riscv64"
    self.convergence_proto = "convergence.prototxt"
    self.caffe_prototxt_for_slipt = "/auto/regrt/sw/titash/llama2/arasw/src/dv/sdk/utils/llama/qwen_converted_omniquant_3bit_2bit_original.prototxt"
    self.qwen_inet = qwen_inet
    self.gatherwts_file_name = "embedded_wts_new_floor_clamped.dat"
    self.config_dir = os.path.abspath(__file__)
    self.config = self.parse_config(config)

    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)


  def get_python_path(self):
    prefix = ':/' + self.run_dir + '/'
    python_path = prefix.join(sys.path)
    container_path = ['/usr/local/lib/python37.zip', '/usr/local/lib/python3.7', '/usr/local/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/site-packages', '/dv/python', '/dv/python/caffe/proto']
    container_path = '/:'.join(container_path)
    return 'PYTHONPATH={}:{}'.format(container_path, python_path)
  
  def write_split_llm_config(self):
    try:
      llama_config = {}
      llama_config["network_file"] = self.caffe_prototxt_for_slipt
      llama_config["model_file"] = "{0}{1}/{2}/{3}/{4}.caffemodel".format(self.run_dir,os.path.abspath(self.config['out']), self.converter_output_dir, self.params_dir, self.converted_name)
      llama_config["prompt_image_quant_path"] = self.config["llama"]["prompt_image_quant_path"]
      llama_config["prompt_image_verif_path"] = self.config["llama"]["prompt_image_verif_path"]
      llama_config["token_image_quant_path"] = self.config["llama"]["token_image_quant_path"]
      llama_config["token_image_verif_path"] = self.config["llama"]["token_image_verif_path"]
      llama_config["prompt_scale_offset_adjust_file"] = "{0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.llama_output_dir,self.prompt_scale_offset_file_name)
      llama_config["token_scale_offset_adjust_file"] = "{0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.llama_output_dir, self.tkn_scale_offset_file_name)
      llama_config["prmpt_prc_inp_max"] = self.config["llama"]["prmpt_prc_inp_max"]
      llama_config["prmpt_prc_inp_min"] = self.config["llama"]["prmpt_prc_inp_min"]
      llama_config["prmpt_prc_inp_stp"] = self.config["llama"]["prmpt_prc_inp_stp"]
      llama_config["tkn_gen_inp_max"] = self.config["llama"]["tkn_gen_inp_max"]
      llama_config["tkn_gen_inp_min"] = self.config["llama"]["tkn_gen_inp_min"]
      llama_config["tkn_gen_inp_stp"] = self.config["llama"]["tkn_gen_inp_stp"]
      llama_config["decoder_start"] = self.config["llama"]["decoder_start"]
      llama_config["decoder_end"] = self.config["llama"]["decoder_end"]
      llama_config["scale_offset_4bit"] = self.config["llama"]["scale_offset_4bit"]
      llama_config["scale_offset_3bit"] = self.config["llama"]["scale_offset_3bit"]
      llama_config["scale_offset_2bit"] = self.config["llama"]["scale_offset_2bit"]
      with open(os.path.join(self.config["out"],self.llama_output_dir,'llm_config.yaml'), 'w') as file:
        yaml.dump(llama_config, file, default_flow_style=False, sort_keys=False)
    except Exception as e:
      logger.error(e)
      sys.exit(1)

  def write_dvm_paths_to_file(self):
    res = []
    for i in os.listdir(os.path.join(self.config["out"],self.llama_output_dir)):
      if not os.path.isdir(os.path.join(self.config["out"],self.llama_output_dir)+'/'+i):
        continue
      k = '0'
      if not i.startswith('prmpt_prc'):
        k = '1'
      tkn_sz = i.split('_')[-1]
      for j in os.listdir(os.path.join(self.config["out"],self.llama_output_dir)+'/'+i):
        if not os.path.isdir(os.path.join(self.config["out"],self.llama_output_dir)+'/'+i+'/'+j):
          continue
        if not j.startswith('split'):
          continue
        idx = j.split('_')[-1]
        res.append(k+','+tkn_sz+','+idx+','+self.config["out"]+'/'+self.llama_output_dir+'/'+i+'/'+j+'/assets/\n')
    with open(os.path.join(self.config["out"],self.llama_output_dir,'all_dvms.txt'), 'w') as f:
      f.writelines(res)
    logger.info("splited model.dvm paths written to : {}".format(os.path.join(self.config["out"],self.llama_output_dir)+'/all_dvms.txt'))
    return res


  def get_rel_tag(self):
    pattern = re.compile("gitea\.kinara\.ai.*sdk\:([rR].*?)\-")
    tag = pattern.search(self.docker_img)
    if tag:
      return tag.group(1)
    else:
      return 'unknown'

  def run_command(self,command, wd, lfn, env=None):
    #print("Running command:", ' '.join(command))
    logger.info(f"Running command : {command} cwd : {wd} log: {lfn}")
    try:
      try:
        lfd = open(lfn, "w+")
      except:
        logger.error("cannot open logfile: %s : %s", lfn, sys.exc_info()[0])
      r = subprocess.run(command, stdout=lfd, stderr=subprocess.STDOUT, env=env, shell=True, text=True, cwd=wd)
      if r.returncode!=0:
        # logger.error(f"failed to run command: {command}")
        raise Exception("command failed with exit code: " + str(r.returncode))
      return r
    except:
      raise Exception(f"error while running the command: {command}")

  def parse_config(self, config):
    logger.info('parsing YAML config file')
    if not os.path.exists(config):
      raise Exception('error config file not accessible - ' + config)
    try:
      local_config = yaml.safe_load(open(config, 'r'))
    except Exception as e:
      raise Exception('error loading config file - ' + str(e))
    self.config_dir = os.path.dirname(os.path.abspath(config))
    logger.info('validating YAML config file')
    if 'preprocess' in local_config and local_config['preprocess']:
      logger.warning('** preprocess option is deprecated, only preprocessed fp32 inputs have to be passed to dvrun **')
    #self.validate_config()
    return local_config

  def get_path(self, path, ignore_error=False):
    if not os.path.isabs(path):
      path = os.path.join(self.config_dir, path)
      if not os.path.exists(path) and not ignore_error:
        raise Exception('error: file not accessible - ' + path)
    elif not os.path.exists(path) and not ignore_error:
      raise Exception('error: file not accessible - ' + path)
    return os.path.abspath(path)

  def set_paths(self):
    if self.config['network']['srcfw'] != 'pytorch':
      self.config['network']['inet'] = self.get_path(self.config['network']['inet'])
    if 'inet_2048' in self.config['network'] and self.config['network']['inet_2048']:
      self.config['network']['inet_2048'] = self.get_path(self.config['network']['inet_2048'])
    if 'pcfg' in self.config['network'] and not self.config['network']['pcfg']:
      self.config['network']['iwt'] = self.get_path(self.config['network']['iwt'])
    else:
      self.config['network']['pcfg'] = self.get_path(self.config['network']['pcfg'])
    if 'tags' in self.config['network'] and self.config['network']['tags']:
      self.config['network']['tags'] = self.get_path(self.config['network']['tags'])
    if 'label' in self.config['network'] and self.config['network']['label']:
      self.config['network']['label'] = self.get_path(self.config['network']['label'])
    # if self.args.mode == 'simulate-only' and 'images' in self.config['dvsim'] and self.config['dvsim']['images']:
    #   self.config['dvsim']['images'] = self.get_path(self.config['dvsim']['images'])
    if 'adjust' in self.config['dvnc'] and self.config['dvnc']['adjust']:
      self.config['dvnc']['adjust'] = self.get_path(self.config['dvnc']['adjust'])
    self.config['network']['images']['quantize'] = self.get_path(self.config['network']['images']['quantize'])
    self.config['network']['images']['verify'] = self.get_path(self.config['network']['images']['verify'])
    self.config['out'] = self.get_path(self.config['out'], True)

  def print_config(self):
    logger.info('release - ' + self.get_rel_tag())
    logger.info('config file:')
    logger.info(yaml.dump(self.config))
    self.create_out_dir(self.config['out'])
    with open(os.path.join(self.config['out'], 'run.yaml'), 'w') as cfgout:
      yaml.dump(self.config, cfgout, default_flow_style=False)
    with open(os.path.join(self.config['out'], 'version'), 'w') as verout:
      verout.write(self.get_rel_tag())

  def create_out_dir(self, directory):
    if not os.path.exists(directory):
      os.makedirs(directory)

  def exec_dvmc(self, dryrun=None):
    DCMD = []
    # vols = self.get_dvconvert_vol_docker(dryrun)
    # env = None
    # if self.config['network']['srcfw'] == 'pytorch':
    #   env = self.get_python_path()
    DCMD.append(self.dvconvert_cmd)
    # if self.args.mode == 'debug':
    #   DCMD.append("-dump_debug 1")
    # else:
    DCMD.append("-dump_debug 0")
    DCMD.append("-src_framework " + self.config['network']['srcfw'])
    DCMD.append("-dst_framework caffe")
    DCMD.append("-input_shape " + self.config['network']['dim'].replace(":", " "))
    DCMD.append("-input_weight {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['iwt'])))
    # if self.args.mode == 'debug':
    #   if self.args.debug_level == 'low':
    #     DCMD.append("-debug 1")
    #   elif self.args.debug_level == 'medium':
    #     DCMD.append("-debug 2")
    #   else:
    #     DCMD.append("-debug 3")
    if dryrun:
      DCMD.append("-output_dir {0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir + str(dryrun), self.converter_output_dir))
      DCMD.append("-summary_csv {0}{1}/{2}/{3}/summary.csv".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir + str(dryrun), self.converter_output_dir))
      DCMD.append("-output_model_name {0}".format(self.dryrun_name))
      # DCMD.append("-debug {0}{1}/{2}/{3}/no_chunks.json".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir1, self.scheduler_output_dir))
      DCMD.append("-hints {0}{1}/{2}/{3}/no_chunks.json".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir1, self.scheduler_output_dir))
      DCMD.append("-dry_run " + str(dryrun))
      DCMD.append("-dynamic False")
      DCMD.append("-dynamic_input False")
    else:
      DCMD.append("-output_dir {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.converter_output_dir))
      DCMD.append("-summary_csv {0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.converter_output_dir, "summary.csv"))
      DCMD.append("-output_model_name {0}".format(self.converted_name))
      # DCMD.append("-debug {0}{1}/{2/{4}/no_chunks.json {0}{1}/{3}/{4}/no_chunks.json".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir1, self.dryrun_dir2, self.scheduler_output_dir))
      DCMD.append("-hints {0}{1}/{2}/{4}/no_chunks.json".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir1, self.dryrun_dir2, self.scheduler_output_dir))
      if 'dynamic' in self.config['dvconvert'] and self.config['dvconvert']['dynamic']:
        DCMD.append("-dynamic True")
      if 'dynamic_input' in self.config['dvconvert'] and self.config['dvconvert']['dynamic_input']:
        DCMD.append("-dynamic_input True")

    if 'lname' in self.config['dvconvert'] and self.config['dvconvert']['lname'] is not None:
      DCMD.append("-label_name " + self.config['dvconvert']['lname'].replace(":", " ").replace("#", ":"))
    if self.config['network']['inet'] is not None and self.config['network']['srcfw'] != "pytorch":
      DCMD.append("-input_network {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['inet'])))
    elif self.config['network']['inet'] is not None and self.config['network']['srcfw'] == "pytorch":
      DCMD.append("-input_network " + self.config['network']['inet'])
    if 'lshape' in self.config['dvconvert'] and self.config['dvconvert']['lshape'] is not None:
      DCMD.append("-label_shape " + self.config['dvconvert']['lshape'].replace(":", " "))
    if 'inode' in self.config['dvconvert'] and self.config['dvconvert']['inode'] is not None:
      DCMD.append("-input_node_name " + self.config['dvconvert']['inode'].replace(":", " ").replace("#", ":"))
    if 'iname' in self.config['dvconvert'] and self.config['dvconvert']['iname'] is not None:
      DCMD.append("-iname " + self.config['dvconvert']['iname'].replace(":", " ").replace("#", ":"))
    if 'onode' in self.config['dvconvert'] and self.config['dvconvert']['onode'] is not None:
      DCMD.append("-output_node_name " + self.config['dvconvert']['onode'].replace(":", " ").replace("#", ":"))
    if 'data_format' in self.config['dvconvert'] and self.config['dvconvert']['data_format'] is not None:
      DCMD.append("-data_format " + self.config['dvconvert']['data_format'].replace(":", " "))
    if 'dformat' in self.config['dvconvert'] and self.config['dvconvert']['dformat'] is not None:
      DCMD.append("-dformat " + self.config['dvconvert']['dformat'].replace(":", " "))
    if 'pcfg' in self.config['network'] and self.config['network']['pcfg'] is not None:
      DCMD.append("-config {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['pcfg'])))
    # DCMD.append("-images {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['images']['verify'])))
    if 'np' in self.config['dvconvert'] and self.config['dvconvert']['np']:
      DCMD.append("-set_np_shape True")
    if 'rsmax' in self.config['dvconvert'] and self.config['dvconvert']['rsmax']:
      DCMD.append("-fork_sftmx_argmx 1")
    if 'ignore_mse' in self.config['dvconvert'] and self.config['dvconvert']['ignore_mse']:
      DCMD.append("-ignore_mse 1")
    if 'state_name' in self.config['dvconvert'] and self.config['dvconvert']['state_name']:
      DCMD.append("-state_name " + self.config['dvconvert']['state_name'].replace(":", " ").replace("#", ":"))
    if 'state_shape' in self.config['dvconvert'] and self.config['dvconvert']['state_shape']:
      DCMD.append("-state_shape " + self.config['dvconvert']['state_shape'].replace(":", " "))
    if 'tg' in self.config['dvconvert'] and self.config['dvconvert']['tg']:
      DCMD.append("-transform_graph True")
    if 'raw_images' in self.config['dvconvert'] and self.config['dvconvert']['raw_images']:
      DCMD.append("-raw_images " + self.config['dvconvert']['raw_images'])
    if 'onnx_backend' in self.config['dvconvert'] and self.config['dvconvert']['onnx_backend']:
      DCMD.append("-onnx_backend " + self.config['dvconvert']['onnx_backend'])
    if 'model_name' in self.config and self.config['model_name']:
      DCMD.append("-name " + self.config['model_name'])
    if not dryrun or dryrun == 2:
      DCMD.append("-fuse_act False")
    if 'dtype' in self.config['dvconvert'] and self.config['dvconvert']['dtype']:
      DCMD.append("-dtype " + self.config['dvconvert']['dtype'])
    if self.config['dvnc']['qmode'] == 9:
      DCMD.append("-platform_target Ara2")
    FINAL_CMD = " ".join(DCMD)
    lfn = os.path.join(os.path.abspath(self.config['out']),"converter.log")
    self.run_command(FINAL_CMD, self.dvconvert_dir, lfn)

  def exec_quantizer(self, dryrun=None):
    DCMD = []
    if dryrun:
      DCMD.append(self.quantizer_cmd)
      DCMD.append("-p {0}{1}/{2}/{3}/{4}/{5}.prototxt".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir + str(dryrun), self.converter_output_dir, self.params_dir, self.dryrun_name))
      DCMD.append("-o {0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir + str(dryrun), self.quantizer_output_dir))
      DCMD.append("-x")
      DCMD.append("-g")
      if self.config['dvnc']['qmode']:
        DCMD.append("-M " + str(self.config['dvnc']['qmode']))
    else:
      DCMD.append(self.quantizer_cmd)
      if self.qwen_inet is not None:
        DCMD.append("-p {0}{1}".format(self.run_dir, os.path.abspath(self.qwen_inet)))
      else:
        DCMD.append("-p {0}{1}/{2}/{3}/{4}.prototxt".format(self.run_dir, os.path.abspath(self.config['out']), self.converter_output_dir, self.params_dir, self.converted_name))
      DCMD.append("-m {0}{1}/{2}/{3}/{4}.caffemodel".format(self.run_dir, os.path.abspath(self.config['out']), self.converter_output_dir, self.params_dir, self.converted_name))
      DCMD.append("-t {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['images']['quantize'])))
      DCMD.append("-v {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['images']['verify'])))
      DCMD.append("-o {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.quantizer_output_dir))
      if 'tags' in self.config['network'] and self.config['network']['tags'] is not None:
        DCMD.append("-z {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['tags'])))
      if 'scale_offset_adjust_file' in self.config['dvnc'] and self.config['dvnc']['scale_offset_adjust_file'] is not None:
        DCMD.append("-G {0}{1}".format(self.run_dir, os.path.abspath(self.config['dvnc']['scale_offset_adjust_file'])))
      if 'label' in self.config['network'] and self.config['network']['label'] is not None:
        DCMD.append("-L {0}{1}".format(self.run_dir, os.path.abspath(self.config['network']['label'])))
      if 'input_type' in self.config['dvnc'] and self.config['dvnc']['input_type'] is not None:
        DCMD.append("-Z " + self.config['dvnc']['input_type'])
      if 'adjust' in self.config['dvnc'] and self.config['dvnc']['adjust'] is not None:
        DCMD.append("-A {0}{1}".format(self.run_dir, os.path.abspath(self.config['dvnc']['adjust'])))
      if self.config['dvnc']['qmode']:
        DCMD.append("-M " + str(self.config['dvnc']['qmode']))
      # if self.args.mode == 'debug':
      #   if self.args.debug_level == 'low':
      #     DCMD.append("-D 1")
      #   elif self.args.debug_level == 'medium':
      #     DCMD.append("-D 2")
      #   else:
      #     DCMD.append("-D 3")

      DCMD.append("-w")
      DCMD.append("-u -f -g")

      if self.config['dvnc']['bc']:
        DCMD.append("-B")
      if 'clampqlimits' in self.config['dvnc'] and self.config['dvnc']['clampqlimits']:
        DCMD.append("-i")
    if self.config['dvnc']['qmode'] == 9:
      DCMD.append("-J {0}/{1}".format(self.conf_dir, "ara2_constraints.json"))
    else:
      DCMD.append("-J {0}/{1}".format(self.conf_dir, "ara1_constraints.json"))
    FINAL_CMD = " ".join(DCMD)
    lfn = os.path.join(os.path.abspath(self.config['out']),"analyzer.log")
    self.run_command(FINAL_CMD, self.dvnc_dir, lfn)

  def exec_gatherwts(self):
    GCMD = []
    GCMD.append(self.gatherwts_cmd)
    if self.config['network']['inet'] is not None:
      GCMD.append("--inet {0}{1}".format(self.run_dir, self.config["network"]["inet"]))
    GCMD.append("--output_file {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.llama_output_dir, self.gatherwts_file_name))
    FINAL_CMD = " ".join(GCMD)
    lfn = os.path.join(os.path.abspath(self.config['out']), self.llama_output_dir,"gatherwts.log")
    self.run_command(FINAL_CMD, self.gatherwts_cmd_dir, lfn)

  def exec_merge_scale_offsets_yaml(self):
    SOCMD = []
    SOCMD.append(self.scale_offset_cmd)
    SOCMD.append("--quantizer_yaml {0}{1}/{2}/{3}/{4}".format(self.run_dir, os.path.abspath(self.config['out']), self.quantizer_output_dir, self.stats_dir, self.quantizer_so_yaml))
    SOCMD.append("--output_dir {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']),self.llama_output_dir))
    if 'scale_offset_adjust_file' in self.config['dvnc'] and self.config['dvnc']['scale_offset_adjust_file'] is not None:
      SOCMD.append("--scale_offset_adjust_file {0}{1}".format(self.run_dir, os.path.abspath(self.config['dvnc']['scale_offset_adjust_file'])))
    lfn = os.path.join(os.path.abspath(self.config['out']), self.llama_output_dir,"merge_scale_offsets_yaml.log")
    FINAL_CMD = " ".join(SOCMD)
    self.run_command(FINAL_CMD, self.scale_offset_cmd_dir, lfn)

  def exec_pre_split(self):
    try:
      env =  os.environ.copy()
      env["PYTHONPATH"] = f"{self.bld_dir}/dv2/python:"+f"{self.bld_dir}/dv2/python/caffe/proto:" + env.get("PYTHONPATH", "")
      SCMD = []
      self.write_split_llm_config()
      SCMD.append(self.split_cmd)
      SCMD.append("--config {0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config["out"]), self.llama_output_dir ,"llm_config.yaml"))
      SCMD.append("--output_dir {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config["out"]),self.llama_output_dir))
      lfn = os.path.join(os.path.abspath(self.config['out']), self.llama_output_dir,"pre.log")
      FINAL_CMD = " ".join(SCMD)
      self.run_command(FINAL_CMD, self.split_cmd_dir, lfn, env =env)
      logger.info("execution of pre is successfully")
    except Exception as e:
      logger.error(str(e))
      logger.warning("execution of pre failed")

  def exec_scheduler(self, dryrun=None):
    DCMD = []
    DCMD.append(self.scheduler_cmd)
    DCMD.append("-c {0}/{1}".format(self.conf_dir, self.config_yaml))
    DCMD.append("-k {0}/{1}".format(self.conf_dir, self.meta_json))
    DCMD.append("-p {0}/{1}".format(self.conf_dir, self.power_config))
    DCMD.append("-j -K -O 5 -m 16 -d -y -R 3 -T -s -q -P")
    if dryrun:
      DCMD.append("-x -G")
      DCMD.append("-i {0}{1}/{2}/{3}/{4}/{5}".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir + str(dryrun), self.quantizer_output_dir, self.params_dir, self.convergence_proto))
      DCMD.append("-o {0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir + str(dryrun), self.scheduler_output_dir))
    else:
      DCMD.append("-i {0}{1}/{2}/{3}/{4}".format(self.run_dir, os.path.abspath(self.config['out']), self.quantizer_output_dir, self.params_dir, self.convergence_proto))
      DCMD.append("-E {0}{1}/{2}/{3}/dv_optimization_error.log".format(self.run_dir, os.path.abspath(self.config['out']), self.dryrun_dir2, self.scheduler_output_dir))
      DCMD.append("-o {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.scheduler_output_dir))
      if self.config['dvnc']['layeroutput']:
        DCMD.append("-M")
    lfn = os.path.join(os.path.abspath(self.config['out']),"estimator.log")
    FINAL_CMD = " ".join(DCMD)
    self.run_command(FINAL_CMD, self.dvnc_dir, lfn)

  def exec_generator(self):
    DCMD = []
    DCMD.append(self.generator_cmd)
    DCMD.append("-a {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.quantizer_output_dir))
    DCMD.append("-e {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.scheduler_output_dir))
    DCMD.append("-c {0}/{1}".format(self.conf_dir, self.config_yaml))
    #DCMD.append("-f {0}/{1}".format(self.conf_dir, self.config_yaml))
    DCMD.append("-b {0}/{1}".format(self.conf_dir, self.kernel_lib))
    DCMD.append("-k {0}/{1}".format(self.conf_dir, self.meta_json))
    DCMD.append("-v {0}/{1}".format(self.conf_dir, self.vpu_conf))
    DCMD.append("-w {0}/{1}/{2}".format(self.conf_dir, self.vpu_conf, self.vpu_meta_json))
    DCMD.append("-o {0}{1}/{2}".format(self.run_dir, os.path.abspath(self.config['out']), self.assets_dir))
    DCMD.append("-i {0}{1}/{2}".format(
        self.run_dir,
        os.path.abspath(self.config['out']),
        self.quantizer_output_dir,
    ))
    lfn = os.path.join(os.path.abspath(self.config['out']),"emitter.log")
    FINAL_CMD = " ".join(DCMD)
    self.run_command(FINAL_CMD, self.dvnc_dir, lfn)

  def exec_sin_cos(self):
    try:
      SCCMD = []
      self.create_out_dir(os.path.join(os.path.abspath(self.config['out']), self.llama_output_dir,self.rope_output_dir))
      SCCMD.append(self.sin_cos_cmd)
      if self.config['network']['inet'] is not None:
        SCCMD.append("--inet {0}{1}".format(self.run_dir, self.config["network"]["inet"]))
      SCCMD.append("--output_dir {0}{1}/{2}/{3}".format(self.run_dir, os.path.abspath(self.config['out']), self.llama_output_dir,self.rope_output_dir))
      SCCMD.append("--output_folder_name {0}".format(self.rope_folder_name))
      SCCMD.append("--scale {0}".format(0.00784313027))
      SCCMD.append("--offset {0}".format(-1))
      lfn = os.path.join(os.path.abspath(self.config['out']), self.llama_output_dir,"sin_cos.log")
      FINAL_CMD = " ".join(SCCMD)
      self.run_command(FINAL_CMD, self.sin_cos_cmd_dir, lfn)
      logger.info("exection of sin_cos successful")
    except Exception as e:
      logger.error(str(e))
      logger.info("exection of sin_cos failed")
      sys.exit(1)

  def exec_llm_model_gen(self):
    try:
      LLMGENCMD = []
      res = self.write_dvm_paths_to_file()
      LLMGENCMD.append(self.llm_model_gen_cmd)
      LLMGENCMD.append("-o {0}{1}".format(self.run_dir, os.path.join(self.config['out'], self.llama_output_dir, 'model.dvm')))
      LLMGENCMD.append("-f {0}{1}".format(self.run_dir, os.path.join(self.config['out'], self.llama_output_dir,'all_dvms.txt')))
      LLMGENCMD.append("-r {0}{1}".format(self.run_dir, os.path.join(self.config['out'], self.llama_output_dir,self.rope_output_dir, self.rope_folder_name)))
      lfn = os.path.join(os.path.abspath(self.config['out']), self.llama_output_dir,"llm_model_gen.log")
      FINAL_CMD = " ".join(LLMGENCMD)
      self.run_command(FINAL_CMD, self.llm_mode_gen_dir, lfn)
      logger.info("execution of llm_mode_gen successfully")
    except Exception as e:
      logger.error(str(e))
      logger.info("execution of llm_model_gen failed")

  def run(self, model_name):
    try:
      self.set_paths()
      self.print_config()
      self.exec_dvmc()
      self.exec_quantizer()
      self.exec_scheduler()
      self.exec_generator()
      # if not self.config['dvnc']['qonly'] and not self.args.skip_simulation:
      logger.info("run executed successfully for {0}".format(model_name))
    except Exception as e:
      logger.error(str(e))
      logger.info("run execution failed for mode: {0}".format(model_name))
      raise Exception("{}".format(model_name))

class llamarun():
  def __init__(self):
    self.args = {}
    self.split_output_path = ""
    self.split_paths = []
    self.jobs = []
    self.error_list = []

  def get_splited_directories(self):

    for yaml_path in glob.glob(self.split_output_path +"/prmpt*/*/"+"dvrun_config.yaml"):
      self.split_paths.append(os.path.dirname(yaml_path))
    

    # uncomment below line if wanted to exclude  prmt/split_32 model
    # self.split_paths = self.split_paths[:-1]

    for yaml_path in glob.glob(self.split_output_path +"/tkn*/*/"+"dvrun_config.yaml"):
      self.split_paths.append(os.path.dirname(yaml_path))
    if len(self.split_paths) < 65:
      logger.error("split  directories are less than 66")
      sys.exit(1)
    if len(self.split_paths) ==0:
      logger.error("no splitted directories found to run")
      sys.exit(1)

  def safe_function_call(self, function_list):
    try:
      function, args = function_list
      return function(args)
    except Exception as e:
      return "ERROR: " + str(e)

  def func_pool_map(self, jobs, max_workers=1):
    try:
      functions_list = jobs
      with multiprocessing.Pool(processes=max_workers) as pool:
        result = pool.map(self.safe_function_call, functions_list)
      for res in result:
        if str(res).startswith("ERROR:"):
          logger.error(res)
          self.error_list.append(str(res))
      return result
    except Exception as e:
      logger.error(str(e))

  def exec_splitted_models_parallel(self):
    num_cores = multiprocessing.cpu_count()
    temp_split_paths = [self.split_paths[32:]]
    for i in temp_split_paths:
      dvobj = dvrun(os.path.join(i, "dvrun_config.yaml"), bld_dir = self.args.build)
      model_name =  "/".join(i.split("/")[-2:]).replace("/","_")
      self.jobs.append([dvobj.run, model_name])
    logger.info("total jobs to run {}".format(len(self.jobs)))
    stime = datetime.datetime.now()
    self.func_pool_map(self.jobs, max_workers = num_cores)
    ftime = datetime.datetime.now()
    # td = self.timediff(ftime, stime)
    # logger.info("time taken to run {} split models : {}h {}m {}s".format(len(self.jobs), td['wc_hrs'], td['wc_mins'], td['wc_secs']))
    logger.info("total_splits: {} passed: {} failed: {}".format(len(self.jobs), len(self.jobs)-len(self.error_list), len(self.error_list)))
    logger.error("failed splits : \n{}".format(self.error_list))
    if len(self.error_list) > 0:
      raise Exception("execution of split models failed")
    # TODO uncomment sys.exit(1) statement once output comparasion between dvsim and analyzer is fixed
    # if len(self.error_list) > 0:
    #   with open(self.triage_path, "w") as file:
    #     for lfn in self.failed_log_list:
    #       file.write("\n".join(self.get_log_tail(lfn)))
    #       file.write("\n")
    #   logger.info("triage: {}".format(self.triage_path))
      # sys.exit(1)

  def run(self):
    dv = dvrun(self.args.config, self.args.qwen_inet, self.args.build)
    dv.set_paths()
    dv.print_config()
    dv.create_out_dir(os.path.join(dv.config["out"],dv.llama_output_dir))
    try:
      dv.exec_dvmc()
    except Exception as e:
      logger.warning(str(e))
    try:
      dv.exec_quantizer()
      dv.exec_gatherwts()
      dv.exec_merge_scale_offsets_yaml()
      dv.exec_pre_split()
      self.split_output_path = os.path.join(dv.config["out"],dv.llama_output_dir)
      self.get_splited_directories()
      self.exec_splitted_models_parallel()
      dv.exec_sin_cos()
      dv.exec_llm_model_gen()
    except Exception as e:
      logger.error(str(e))
      sys.exit(1)


  def parse_args(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to configuration file')
    parser.add_argument('--build', required=True, help='build path')
    parser.add_argument('--qwen_inet', required=False, default="", help='qwen proto path')
    self.args = parser.parse_args()
    # if not self.args.mount_config:
    #   self.args.mount_config = self.args.config
    logger.info("parsed command line arguments")

if __name__ == '__main__':
  runner = llamarun()
  runner.parse_args()
  runner.run()
        