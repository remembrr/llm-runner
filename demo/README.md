# Demo

`demo.gif` in the top-level README is a recording of the real `llm-runner` binary —
`llm-runner -h`, then a chat completion.

**The model is pinned and SHA-verified** so the demo reproduces the same *content*:
`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, quant `Q4_K_M`, SHA-256 `74a4da8c…`.
`fetch-model.sh` downloads it via llama.cpp if it isn't cached and verifies the hash
(failing loudly if upstream ever changes the file). The demo runs greedy (`-t 0`), so
the pinned weights produce the **same model output** every time.

The exact GIF *bytes* are not pinned — they also depend on the `asciinema`/`agg`
versions and the recording/render environment. What's reproducible is the setup: the
same commands, the same verified model, and the same answer.

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
