#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm> 
#include <cxxopts.hpp>
#include "llm_params_struct.h"
#include "ClArgs.h"



const long long DDR_SIZE = 1ULL << 34;
#define DV__DDR__BASE  0x080000000ULL
static char ToHex(int ch) {
  return ch < 10 ? static_cast<char>('0' + ch)
                 : static_cast<char>('a' + (ch - 10));
};

unsigned int crc32h(const unsigned char *message, unsigned int size) {
  int i, crc;
  unsigned int byte, c;
  const unsigned int g0 = 0xEDB88320, g1 = g0 >> 1, g2 = g0 >> 2, g3 = g0 >> 3, g4 = g0 >> 4, g5 = g0 >> 5, g6 = (g0 >> 6) ^ g0, g7 = ((g0 >> 6) ^ g0) >> 1;

  i = 0;
  crc = 0xFFFFFFFF;
  for (int i = 0; i < size; i++) {
    byte = message[i];
    crc = crc ^ byte;
    c = ((crc << 31 >> 31) & g7) ^ ((crc << 30 >> 31) & g6) ^ ((crc << 29 >> 31) & g5) ^ ((crc << 28 >> 31) & g4) ^ ((crc << 27 >> 31) & g3) ^ ((crc << 26 >> 31) & g2) ^ ((crc << 25 >> 31) & g1) ^ ((crc << 24 >> 31) & g0);
    crc = ((unsigned)crc >> 8) ^ c;
  }

  return ~crc;
}

struct contents{
  contents(int a, int b, std::string c){
    tkn_sz = a;
    idx = b;
    path = c;
  };
  int tkn_sz;
  int idx;
  std::string path;
};

void get_all_paths(std::string path_txt, std::vector<contents> &all_dvms, int mode){
  std::ifstream file(path_txt);
  std::string str; 
  while (std::getline(file, str)){
    if (str.size() > 3){
      if (str[-1] == '\n') str.pop_back();
      int a, b, z=-1;
      int c=0, e=0;
      for(int d=0; d<str.size(); d++){
        if ((e==3)&&(z==mode)) {
          all_dvms.push_back(contents(a,b,str.substr(c,str.size()-c)));
          break;
        }
        if(str[d]==','){
          if (e==0) z = std::stoi(str.substr(c,d-c+1));
          else if (e==1) a = std::stoi(str.substr(c,d-c+1));
          else if (e==2) b = std::stoi(str.substr(c,d-c+1));
          c = d+1;
          e++;
        }
      }
    }
  }
};

std::string get_dst_path(std::string &path){
  int j = 0, c = 0;
  for(int i=path.size()-1; i>-1; i--){
    if(c<5){
      if(path[i]=='/') c++;
    }else{
      j = i;
      break;
    };
  };
  return path.substr(0,j+1);
};

void get_all_indices(std::vector<contents> &all_dvms, std::vector<int>& vec, std::map<int,int>& indices){
  std::set<int> set1;
  for(auto i: all_dvms){
    set1.insert(i.tkn_sz);
  }
  for(auto i: set1){
    vec.push_back(i);
  }
  std::sort(vec.begin(), vec.end());
  for(int i=0; i<vec.size(); i++){
    indices[vec[i]] = i;
  }
}

long next4kaddr(long addr) {
  long pagesize = 4096;
  return (addr + pagesize) & (~(pagesize - 1));
}

int loadBin(std::string path, uint64_t &addr) {
    struct stat st;
    FILE *fp;
    int err = 0;
    int ret;
    long size;

    const char* path_c = path.c_str();
    fp = fopen(path_c, "r");

    ret = stat(path_c, &st);

    if (st.st_size) {
      ret = fread((void *)addr, 1, st.st_size, fp);
    }
    addr = next4kaddr(addr + st.st_size); 
    if (fp) fclose(fp);
    return st.st_size;
}


int construct_fbs2_model(std::string path_txt, std::string output_dvm, std::string rope_data_path) {
  // load the model.dvm s
  std::vector<contents> all_dvms_p, all_dvms_t;
  get_all_paths(path_txt, all_dvms_p, 0);
  get_all_paths(path_txt, all_dvms_t, 1);
  // std::string dst_path = "/auto/worka/shubham/g/s3/arasw/sw1/candid/test/networks/dv_cfg0/caffe/modelzoo/llama2_no_embeddings_splits/combined_1";
  std::string dst_path = get_dst_path(all_dvms_p[0].path);
  std::vector<int> tkn_sizes, cntx_sizes;
  std::map<int,int> indices_tkn, indices_cntx;
  // TODO: take step_size as input
  int step_size = 256;
  get_all_indices(all_dvms_p, tkn_sizes, indices_tkn);
  get_all_indices(all_dvms_t, cntx_sizes, indices_cntx);
  std::string amap_file = dst_path + "/combined_66.dvm";
  if (output_dvm!="")
        amap_file = output_dvm;
  uint64_t ddr_addr = DVM_DDR_BASE_ADDR;

  int amap_fd = open(amap_file.c_str(), O_CREAT | O_RDWR, 0660);
  uint64_t max_ddr_size = 16*1024*1024*1024ULL;

  int err = ftruncate(amap_fd, max_ddr_size);

  uint8_t *ddr = (uint8_t *)mmap((void *)ddr_addr, max_ddr_size, PROT_READ | PROT_WRITE, MAP_SHARED, amap_fd, 0);
  llm_params_t *llm_params = (llm_params_t*)ddr_addr;
  ddr_addr = next4kaddr(ddr_addr + sizeof(llm_params_t));

  for(int k =0; k < CTX_SIZE; k++) {
      llm_params->prompt_indices[k] = (k-1)/step_size;
      llm_params->token_indices[k] = (k-1)/step_size;
  }
  std::map<int, std::pair<int64_t,int>> dict_klib, dict_qpcmd;
  for (int k=0; k<all_dvms_p.size(); k++){
      contents tmp = all_dvms_p[k];
      // filling params
      llm_params->prompt_params_addr[indices_tkn[tmp.tkn_sz]][tmp.idx] = ddr_addr;
      loadBin(tmp.path + "/params.bin", ddr_addr);
      // filling weights
      llm_params->prompt_weights_addr[tmp.idx] = ddr_addr;
      loadBin(tmp.path + "/rearranged_weights.bin", ddr_addr);
      auto last_addr = ddr_addr;
      auto last_size = loadBin(tmp.path + "/PPA.json", ddr_addr);
      auto value = crc32h((const unsigned char *)last_addr, last_size);
      // std::cout<<"P "<<value<<" "<<tmp.path + "/PPA.json\n";
      if (dict_qpcmd.count(value)==0){
          // filling kernel lib
          ddr_addr = last_addr;
          last_size = loadBin(tmp.path + "/kernel.lib", ddr_addr);
          dict_klib[value] = std::pair<int64_t,int>{last_addr, last_size};
          llm_params->prompt_kernel_lib_addr[indices_tkn[tmp.tkn_sz]][tmp.idx] = last_addr;
          // filling qpcmds
          last_addr = ddr_addr;
          last_size = loadBin(tmp.path + "/qp2.cmds.bin", ddr_addr);
          dict_qpcmd[value] = std::pair<int64_t,int>{last_addr, last_size};
          llm_params->prompt_cmd_addr[indices_tkn[tmp.tkn_sz]][tmp.idx] = last_addr;
          llm_params->prompt_cmd_size[indices_tkn[tmp.tkn_sz]][tmp.idx] = last_size;
      }else{
          llm_params->prompt_kernel_lib_addr[indices_tkn[tmp.tkn_sz]][tmp.idx] = dict_klib[value].first;
          llm_params->prompt_cmd_addr[indices_tkn[tmp.tkn_sz]][tmp.idx] = dict_qpcmd[value].first;
          llm_params->prompt_cmd_size[indices_tkn[tmp.tkn_sz]][tmp.idx] = dict_qpcmd[value].second;
          ddr_addr = last_addr;
      }
  } 
  dict_klib.clear();
  dict_qpcmd.clear();
  for (int k=0; k<all_dvms_t.size(); k++){
      contents tmp = all_dvms_t[k];
      // filling params
      llm_params->token_params_addr[indices_cntx[tmp.tkn_sz]][tmp.idx] = ddr_addr;
      loadBin(tmp.path + "/params.bin", ddr_addr);
      // filling weights
      llm_params->token_weights_addr[tmp.idx] = ddr_addr;
      loadBin(tmp.path + "/rearranged_weights.bin", ddr_addr);
      auto last_addr = ddr_addr;
      auto last_size = loadBin(tmp.path + "/PPA.json", ddr_addr);
      auto value = crc32h((const unsigned char *)last_addr, last_size);
      // std::cout<<"T "<<value<<" "<<tmp.path + "/PPA.json\n";
      if (dict_qpcmd.count(value)==0){
          // filling kernel lib
          ddr_addr = last_addr;
          last_size = loadBin(tmp.path + "/kernel.lib", ddr_addr);
          dict_klib[value] = std::pair<int64_t,int>{last_addr, last_size};
          llm_params->token_kernel_lib_addr[indices_cntx[tmp.tkn_sz]][tmp.idx] = last_addr;
          // filling qpcmds
          last_addr = ddr_addr;
          last_size = loadBin(tmp.path + "/qp2.cmds.bin", ddr_addr);
          dict_qpcmd[value] = std::pair<int64_t,int>{last_addr, last_size};
          llm_params->token_cmd_addr[indices_cntx[tmp.tkn_sz]][tmp.idx] = last_addr;
          llm_params->token_cmd_size[indices_cntx[tmp.tkn_sz]][tmp.idx] = last_size;
      }else{
          llm_params->token_kernel_lib_addr[indices_cntx[tmp.tkn_sz]][tmp.idx] = dict_klib[value].first;
          llm_params->token_cmd_addr[indices_cntx[tmp.tkn_sz]][tmp.idx] = dict_qpcmd[value].first;
          llm_params->token_cmd_size[indices_cntx[tmp.tkn_sz]][tmp.idx] = dict_qpcmd[value].second;
          ddr_addr = last_addr;
      }
  } 
  // copying token data in prompt for 32nd layer
  for(int ii=0; ii<TOKENS; ii++){
      llm_params->prompt_kernel_lib_addr[ii][32] = llm_params->token_kernel_lib_addr[ii][32];
      llm_params->prompt_params_addr[ii][32] = llm_params->token_params_addr[ii][32];
      llm_params->prompt_cmd_addr[ii][32] = llm_params->token_cmd_addr[ii][32];
      llm_params->prompt_cmd_size[ii][32] = llm_params->token_cmd_size[ii][32];
  }
  llm_params->prompt_weights_addr[32] = llm_params->token_weights_addr[32];
  // filling other data
  llm_params->prompt_rope_addr[0] = ddr_addr;
  loadBin(rope_data_path + "/rope_256x128_int8.dat", ddr_addr);
  llm_params->prompt_rope_addr[1] = ddr_addr;
  loadBin(rope_data_path + "/rope_256x256_int8.dat", ddr_addr);
  llm_params->prompt_rope_addr[2] = ddr_addr;
  loadBin(rope_data_path + "/rope_256x384_int8.dat", ddr_addr);
  llm_params->prompt_rope_addr[3] = ddr_addr;
  loadBin(rope_data_path + "/rope_256x512_int8.dat", ddr_addr);
  llm_params->token_rope_addr = ddr_addr;
  loadBin(rope_data_path + "/rope_2048x256_int8.dat", ddr_addr);
  uint64_t final_size = ddr_addr - DVM_DDR_BASE_ADDR;
  // std::cout<<"line 100 "<<ddr_addr<<"\n";
  llm_params->num_levels = indices_tkn.size();
  llm_params->num_layers = all_dvms_p.size()/indices_tkn.size();
  llm_params->step_size = step_size;
  llm_params->l2m_pos_addr = 0x12000000;
  int input_size = CTX_SIZE*HIDDEN_SIZE*IN_PRECISION/8;
  llm_params->input_addr = ddr_addr;
  ddr_addr = next4kaddr(ddr_addr + input_size); 
  int output_size = CTX_SIZE*HIDDEN_SIZE*IN_PRECISION/8;
  llm_params->output_addr = ddr_addr;
  ddr_addr = next4kaddr(ddr_addr + output_size); 
  llm_params->final_output_addr = ddr_addr;
  ddr_addr = next4kaddr(ddr_addr + VOCAB_SIZE*2); 
  llm_params->k_addr = ddr_addr;
  ddr_addr = next4kaddr(ddr_addr + CTX_SIZE*HIDDEN_SIZE*(KV_PRECISION/8)*TOTAL_DECODER_LAYERS);
  llm_params->v_addr = ddr_addr;
  ddr_addr = next4kaddr(ddr_addr + CTX_SIZE*HIDDEN_SIZE*(KV_PRECISION/8)*TOTAL_DECODER_LAYERS);
  llm_params->scratch_addr = ddr_addr;
  // ddr_addr = next4kaddr(ddr_addr + 64*1024*1024);


  std::ofstream myfile (amap_file+".txt");
  if (myfile.is_open()){
      myfile << std::hex<< "num_layers: "<<llm_params->num_layers<<"\n";
      myfile << "num_levels: "<<llm_params->num_levels<<"\n";
      myfile << "step_size: "<<llm_params->step_size<<"\n";
      myfile << "input_addr: 0x"<<llm_params->input_addr<<"\n";
      myfile << "output_addr: 0x"<<llm_params->output_addr<<"\n";
      myfile << "scratch_addr: 0x"<<llm_params->scratch_addr<<"\n";
      myfile << "final_output_addr: 0x"<<llm_params->final_output_addr<<"\n";
      myfile << "k_addr: 0x"<<llm_params->k_addr<<"\n";
      myfile << "v_addr: 0x"<<llm_params->v_addr<<"\n";
      myfile << "l2m_pos_addr: 0x"<<llm_params->l2m_pos_addr<<"\n";
      myfile << "prompt_rope_addr:\n";
      for (int jj=0; jj<TOKENS; jj++){
          if (llm_params->prompt_rope_addr[jj] != 0)
              myfile<<jj<<": 0x"<<llm_params->prompt_rope_addr[jj]<<"\n";
      }
      myfile << "token_rope_addr:\n";
      myfile<<llm_params->token_rope_addr<<"\n";
      myfile << "prompt_indices:\n";
      for (int jj=0; jj<CTX_SIZE; jj++){
          if (llm_params->prompt_indices[jj] != 0)
              myfile<<jj<<": "<<llm_params->prompt_indices[jj]<<", ";
      }
      myfile << "\ntoken_indices:\n";
      for (int jj=0; jj<CTX_SIZE; jj++){
          if (llm_params->token_indices[jj] != 0)
              myfile<<jj<<": "<<llm_params->token_indices[jj]<<", ";
      }
      myfile << "\nprompt_kernel_lib_addr:\n";
      for (int ii=0; ii<TOKENS; ii++){
          for (int jj=0; jj<SPLITS; jj++){
              if (llm_params->prompt_kernel_lib_addr[ii][jj] != 0)
                  myfile <<ii<<","<<jj<<": 0x"<<llm_params->prompt_kernel_lib_addr[ii][jj]<<"\n";
          }
      }
      myfile << "prompt_params_addr:\n";
      for (int ii=0; ii<TOKENS; ii++){
          for (int jj=0; jj<SPLITS; jj++){
              if (llm_params->prompt_params_addr[ii][jj] != 0)
                  myfile <<ii<<","<<jj<<": 0x"<<llm_params->prompt_params_addr[ii][jj]<<"\n";
          }
      }
      myfile << "prompt_cmd_addr & prompt_cmd_size:\n";
      for (int ii=0; ii<TOKENS; ii++){
          for (int jj=0; jj<SPLITS; jj++){
              if (llm_params->prompt_cmd_addr[ii][jj] != 0)
                  myfile <<ii<<","<<jj<<": 0x"<<llm_params->prompt_cmd_addr[ii][jj]<<" "<<llm_params->prompt_cmd_size[ii][jj]<<"\n";
          }
      }
      myfile << "prompt_weights_addr:\n";
      for (int jj=0; jj<SPLITS; jj++){
          if (llm_params->prompt_weights_addr[jj] != 0)
              myfile <<jj<<": 0x"<<llm_params->prompt_weights_addr[jj]<<"\n";
      }
      myfile << "token_kernel_lib_addr:\n";
      for (int ii=0; ii<TOKENS; ii++){
          for (int jj=0; jj<SPLITS; jj++){
              if (llm_params->token_kernel_lib_addr[ii][jj] != 0)
                  myfile <<ii<<","<<jj<<": 0x"<<llm_params->token_kernel_lib_addr[ii][jj]<<"\n";
          }
      }
      myfile << "token_params_addr:\n";
      for (int ii=0; ii<TOKENS; ii++){
          for (int jj=0; jj<SPLITS; jj++){
              if (llm_params->token_params_addr[ii][jj] != 0)
                  myfile <<ii<<","<<jj<<": 0x"<<llm_params->token_params_addr[ii][jj]<<"\n";
          }
      }
      myfile << "token_cmd_addr & token_cmd_size:\n";
      for (int ii=0; ii<TOKENS; ii++){
          for (int jj=0; jj<SPLITS; jj++){
              if (llm_params->token_cmd_addr[ii][jj] != 0)
                  myfile <<ii<<","<<jj<<": 0x"<<llm_params->token_cmd_addr[ii][jj]<<" "<<llm_params->token_cmd_size[ii][jj]<<"\n";
          }
      }
      myfile << "token_weights_addr:\n";
      for (int jj=0; jj<SPLITS; jj++){
          if (llm_params->token_weights_addr[jj] != 0)
              myfile<<jj<<": 0x"<<llm_params->token_weights_addr[jj]<<"\n";
      }
      myfile.close();
  }
  err = ftruncate(amap_fd, final_size);
  if(ddr) munmap(ddr, max_ddr_size);
  close(amap_fd);

  return 0;
};

int main(int argc, char **argv){
  auto args = parse_cli_args(argc, argv);
  auto output_dvm = args["output_dvm"].as<std::string>();
  auto path_txt = args["path_txt"].as<std::string>();
  auto rope_data_path = args["rope_data_path"].as<std::string>();

  construct_fbs2_model(path_txt, output_dvm, rope_data_path);
  return 0;
}

 
