#pragma once

struct llama_context_params {
    float defrag_thold;
    unsigned int n_ctx;
};

struct llama_token_data {
    int id;
    float logit;
};

struct llama_sampler_chain_params {
    int no_perf;
};

enum ggml_prec {
    GGML_PREC_UNDEFINED = 0,
    GGML_PREC_DEFAULT = 0,
};
