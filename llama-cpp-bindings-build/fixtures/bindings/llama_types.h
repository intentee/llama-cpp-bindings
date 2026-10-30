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
