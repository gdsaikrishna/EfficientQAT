#define SPLITS 36
#define TOKENS 8
#define CTX_SIZE 2048
#define HIDDEN_SIZE 4096
#define IN_PRECISION 16
#define VOCAB_SIZE 152000
#define KV_PRECISION 8
#define TOTAL_DECODER_LAYERS 32
#define DVM_DDR_BASE_ADDR 0x90000000ULL

typedef struct __attribute__((packed)){
  uint32_t num_layers;
  uint32_t num_levels; // number of different resolutions of cmd stream
  uint32_t step_size;
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t scratch_addr;
  uint64_t final_output_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t l2m_pos_addr;
  uint64_t prompt_rope_addr[TOKENS];
  uint64_t token_rope_addr;
  uint32_t prompt_indices[CTX_SIZE];
  uint32_t token_indices[CTX_SIZE];
  uint64_t prompt_kernel_lib_addr[TOKENS][SPLITS];
  uint64_t token_kernel_lib_addr[TOKENS][SPLITS];
  uint64_t prompt_params_addr[TOKENS][SPLITS];
  uint64_t token_params_addr[TOKENS][SPLITS];
  uint64_t prompt_cmd_addr[TOKENS][SPLITS];
  uint32_t prompt_cmd_size[TOKENS][SPLITS];
  uint64_t token_cmd_addr[TOKENS][SPLITS];
  uint32_t token_cmd_size[TOKENS][SPLITS];
  uint64_t prompt_weights_addr[SPLITS];
  uint64_t token_weights_addr[SPLITS];
} llm_params_t;
