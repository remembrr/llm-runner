# Demo

`demo.gif` in the top-level README is a recording of the real `llm-runner` binary —
`llm-runner -h`, then a chat completion.

**The model is pinned and SHA-verified** so the demo reproduces the same *content*:
`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, quant `Q4_K_M`, SHA-256 `74a4da8c…`.
`fetch-model.sh` downloads it via llama.cpp if it isn't cached and verifies the hash
(failing loudly if upstream ever changes the file). The demo runs greedy (`-t 0`), so it **reproduces the validated demo content
in the documented environment**. Greedy decoding is deterministic given identical
logits, but a different llama.cpp build or hardware backend (Metal / CPU / CUDA) can
still shift the output — so the same answer is not guaranteed across environments.

The exact GIF *bytes* are not pinned either — they also depend on the `asciinema`/`agg`
versions and the recording/render environment. What's pinned is the setup: the same
commands and the same SHA-verified model.

## Regenerate

Needs `asciinema` and `agg` (`brew install asciinema agg`), and a built binary
(`cmake --build build`). From the repo root:

```sh
MODEL=$(sh demo/fetch-model.sh)
asciinema rec --overwrite -c "sh demo/record.sh $MODEL" demo/demo.cast
agg demo/demo.cast demo/demo.gif
```

- `fetch-model.sh` — pins + downloads + SHA-verifies the model; prints its path.
- `record.sh` — drives the recorded session (takes the model path).
- `demo.cast` — the asciinema source; `demo.gif` — the rendered result.
