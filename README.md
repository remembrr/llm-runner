# llm-runner

[![build](https://github.com/remembrr/llm-runner/actions/workflows/build.yml/badge.svg)](https://github.com/remembrr/llm-runner/actions/workflows/build.yml)

A minimal C++ wrapper around **libllama** (llama.cpp). It links the installed
llama.cpp library, loads a GGUF model, and streams a completion from a prompt.

It is a starting point for building your own in-process tool — it does *not*
re-implement inference. The engine, GPU backends, and GGUF handling all come
from llama.cpp. For a ready-made CLI, server, and web UI, use `llama-cli` /
`llama-server` directly; build this only when you need custom in-process logic.

## Build

Verified against llama.cpp **v1.0.0 (build 12359)** installed via Homebrew.

```bash
cmake -B build          # defaults LLAMA_PREFIX to /opt/homebrew/opt/llama.cpp
cmake --build build -j
```

On another machine, point it at your install:

```bash
cmake -B build -DLLAMA_PREFIX=$(brew --prefix llama.cpp)
# or, against a from-source build you installed:
cmake -B build -DLLAMA_PREFIX=/path/to/llama.cpp/install
```

## Run

By default the prompt is a **user turn wrapped in the model's chat template**
(from GGUF metadata), so instruct/chat models work directly:

```bash
./build/llm-runner -m model.gguf -sys "You are terse." -p "Why is the sky blue?" -n 60
```

Interactive multi-turn chat (`-i`) keeps conversation history and reuses the KV
cache across turns — only each new turn's tokens are decoded. When history would
overflow `-c`, it slides the window: the oldest turns are dropped (the system
prompt and current turn are kept) and it continues instead of stopping. `/clear`
resets, `/exit` or Ctrl-D quits:

```bash
./build/llm-runner -i -m model.gguf -sys "You are a helpful assistant."
```

For base/completion models, `--raw` skips the template and continues the prompt:

```bash
./build/llm-runner -m base-model.gguf --raw -p "Once upon a time" -n 40
```

Options: `-m <gguf>` (required), `-p <prompt>`, `-sys <system prompt>`,
`-i`/`--interactive`, `--raw`, `-n <tokens per reply>`, `-c <ctx>`,
`-ngl <gpu-layers>` (99 = all, Metal on Apple Silicon), `-t <temperature>`
(0 = greedy), `-h`.

## Next steps

- **Context when full:** interactive mode slides the window — it drops the oldest
  user+assistant pairs (keeping the system prompt and current turn) until the render
  fits `-c`, then continues. Dropped turns are gone, not summarized; for long-horizon
  memory, replace the hard drop with a summarize-and-prepend step, or add retrieval.
  A single turn larger than the whole context is skipped with a notice (raise `-c`).
- **Sampling:** fixed top-k/top-p/temp chain. Add repetition penalties
  (`llama_sampler_init_penalties`), min-p, or a grammar (`llama_sampler_init_grammar`)
  to the chain in `main()` as needed.
- **Pull from Hugging Face:** this takes a local `-m` path. To download by repo
  like the CLI does, run `llama-cli -hf user/repo:quant` once; it caches under the
  HF hub layout, e.g.
  `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<hash>/<file>.gguf`.
  Point `-m` at that file, or link llama.cpp's `common` download helpers to do it
  in-process.
- **Serve an API / web UI:** don't rebuild these — `llama-server` already exposes
  an OpenAI-compatible API and a web chat UI.
