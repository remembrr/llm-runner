// llm-runner — minimal C++ wrapper around libllama (llama.cpp).
// Modes:
//   single-shot (default): one prompt -> one completion. Chat-templated unless --raw.
//   interactive (-i):      a REPL that keeps conversation history across turns,
//                          reusing the KV cache (only new tokens are decoded each turn).
//
// Built and verified against llama.cpp build 12359 (v1.0.0) via the C API in llama.h.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [options]\n"
        "  -m   <path>    GGUF model file (required)\n"
        "  -p   <text>    prompt / first user turn (default: \"Hello, my name is\")\n"
        "  -sys <text>    system prompt\n"
        "  -i, --interactive   multi-turn chat REPL (keeps history; /clear, /exit)\n"
        "  --raw          raw completion: skip the chat template (single-shot only)\n"
        "  -n   <int>     max tokens per reply (default: 128)\n"
        "  -c   <int>     context size      (default: 4096)\n"
        "  -ngl <int>     GPU layers to offload, 99 = all (default: 99, Metal on Apple Silicon)\n"
        "  -t   <float>   temperature, 0 = greedy (default: 0.7)\n"
        "  -h             this help\n",
        prog);
}

static void quiet_log(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// --- small helpers over the C API ---

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &s, bool add_special) {
    const int n = -llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> toks(n > 0 ? n : 0);
    if (n > 0) {
        llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), toks.data(), (int32_t) toks.size(), add_special, true);
    }
    return toks;
}

static std::string token_piece(const llama_vocab *vocab, llama_token id) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, /*special*/ true);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}

// Render messages with the model's chat template (two-pass: size, then fill).
static std::string render_chat(const char *tmpl,
                               const std::vector<std::pair<std::string, std::string>> &msgs,
                               bool add_assistant) {
    std::vector<llama_chat_message> cm;
    cm.reserve(msgs.size());
    for (const auto &m : msgs) cm.push_back({ m.first.c_str(), m.second.c_str() });

    const int32_t need = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_assistant, nullptr, 0);
    if (need <= 0) return std::string();
    std::vector<char> buf((size_t) need);
    const int32_t got = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_assistant,
                                                  buf.data(), (int32_t) buf.size());
    return got > 0 ? std::string(buf.data(), (size_t) got) : std::string();
}

static size_t common_prefix(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    return n;
}

// Decode a run of tokens, then sample/generate up to n_predict, streaming to stdout.
// Keeps `processed` (the tokens currently in the KV cache) in lockstep with what is decoded,
// and returns the assistant's reply text.
static bool generate(llama_context *ctx, const llama_vocab *vocab, llama_sampler *smpl,
                     std::vector<llama_token> to_decode, int n_predict,
                     std::vector<llama_token> &processed, std::string &reply_out) {
    llama_token id = 0;
    int generated = 0;
    std::vector<llama_token> cur = std::move(to_decode);

    while (true) {
        llama_batch batch = llama_batch_get_one(cur.data(), (int32_t) cur.size());
        if (llama_decode(ctx, batch)) { fprintf(stderr, "\n[error: llama_decode failed]\n"); return false; }
        for (llama_token t : cur) processed.push_back(t);   // now in the KV cache

        id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        const std::string p = token_piece(vocab, id);
        fputs(p.c_str(), stdout);
        fflush(stdout);
        reply_out += p;

        if (++generated >= n_predict) {
            // decode this final token too, so `processed` stays == KV contents
            llama_batch last = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, last) == 0) processed.push_back(id);
            break;
        }
        cur.assign(1, id);
    }
    return true;
}

static int run_interactive(llama_context *ctx, const llama_vocab *vocab, llama_sampler *smpl,
                           const char *tmpl, const std::string &system_prompt,
                           int n_predict, int n_ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<std::pair<std::string, std::string>> messages;
    if (!system_prompt.empty()) messages.push_back({ "system", system_prompt });
    std::vector<llama_token> processed;   // tokens currently in the KV cache (seq 0)

    fprintf(stderr, "Interactive chat — type a message; /clear resets history, /exit (or Ctrl-D) quits.\n");

    std::string line;
    while (true) {
        fputs("\n> ", stdout);
        fflush(stdout);
        if (!std::getline(std::cin, line)) { fputc('\n', stdout); break; }   // EOF
        if (line == "/exit" || line == "/quit") break;
        if (line == "/clear") {
            messages.clear();
            if (!system_prompt.empty()) messages.push_back({ "system", system_prompt });
            llama_memory_clear(mem, true);
            processed.clear();
            fprintf(stderr, "[history cleared]\n");
            continue;
        }
        if (line.empty()) continue;

        messages.push_back({ "user", line });

        // Keep the system message (if any) pinned; the current user turn is always last.
        const size_t sys_off = (!messages.empty() && messages.front().first == "system") ? 1 : 0;
        auto rebuild = [&]() {
            return tokenize(vocab, render_chat(tmpl, messages, /*add_assistant*/ true), /*add_special*/ false);
        };
        std::vector<llama_token> toks = rebuild();

        // Sliding window: while it won't fit, evict the oldest turn (a user+assistant
        // pair), never the system message or the current user turn, then re-render.
        bool evicted = false;
        while ((int) toks.size() + n_predict >= n_ctx && messages.size() > sys_off + 1) {
            messages.erase(messages.begin() + (long) sys_off);                 // oldest user turn
            if (messages.size() > sys_off + 1) {
                messages.erase(messages.begin() + (long) sys_off);             // its assistant reply
            }
            toks = rebuild();
            evicted = true;
        }
        if ((int) toks.size() + n_predict >= n_ctx) {
            fprintf(stderr, "[this turn alone exceeds the %d-token context; skipped — raise -c]\n", n_ctx);
            messages.pop_back();
            continue;
        }
        if (evicted) {
            fprintf(stderr, "[context full — dropped oldest turns to fit %d tokens]\n", n_ctx);
        }

        // Reuse the KV cache for the shared prefix; drop only the diverged tail.
        const size_t ncommon = common_prefix(processed, toks);
        if (ncommon < processed.size()) {
            llama_memory_seq_rm(mem, 0, (llama_pos) ncommon, -1);
            processed.resize(ncommon);
        }

        std::vector<llama_token> suffix(toks.begin() + (long) ncommon, toks.end());
        std::string reply;
        if (!generate(ctx, vocab, smpl, std::move(suffix), n_predict, processed, reply)) break;
        fputc('\n', stdout);
        messages.push_back({ "assistant", reply });
    }
    return 0;
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "Hello, my name is";
    std::string system_prompt;
    bool  raw          = false;
    bool  interactive  = false;
    int   n_predict    = 128;
    int   n_ctx        = 4096;
    int   n_gpu_layers = 99;
    float temperature  = 0.7f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")   model_path   = need("-m");
        else if (a == "-p")   prompt       = need("-p");
        else if (a == "-sys") system_prompt = need("-sys");
        else if (a == "-n")   n_predict    = atoi(need("-n"));
        else if (a == "-c")   n_ctx        = atoi(need("-c"));
        else if (a == "-ngl") n_gpu_layers = atoi(need("-ngl"));
        else if (a == "-t")   temperature  = (float) atof(need("-t"));
        else if (a == "--raw") raw = true;
        else if (a == "-i" || a == "--interactive") interactive = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: failed to load model: %s\n", model_path.c_str()); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) n_ctx;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); llama_model_free(model); return 1; }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    int rc = 0;
    if (interactive) {
        const char *tmpl = llama_model_chat_template(model, nullptr);
        if (!tmpl) {
            fprintf(stderr, "error: -i needs a chat template, but this model has none.\n");
            rc = 1;
        } else {
            rc = run_interactive(ctx, vocab, smpl, tmpl, system_prompt, n_predict, n_ctx);
        }
    } else {
        // --- single-shot ---
        std::string input = prompt;
        bool add_special = true;
        if (!raw) {
            const char *tmpl = llama_model_chat_template(model, nullptr);
            if (!tmpl) {
                fprintf(stderr, "warning: model has no chat template; falling back to --raw\n");
            } else {
                std::vector<std::pair<std::string, std::string>> msgs;
                if (!system_prompt.empty()) msgs.push_back({ "system", system_prompt });
                msgs.push_back({ "user", prompt });
                input = render_chat(tmpl, msgs, /*add_assistant*/ true);
                if (input.empty()) { fprintf(stderr, "error: chat template apply failed\n"); rc = 1; }
                add_special = false;
            }
        }

        if (rc == 0) {
            std::vector<llama_token> toks = tokenize(vocab, input, add_special);
            if (raw) { fputs(prompt.c_str(), stdout); fflush(stdout); }
            std::vector<llama_token> processed;
            std::string reply;
            generate(ctx, vocab, smpl, std::move(toks), n_predict, processed, reply);
            fputc('\n', stdout);
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
