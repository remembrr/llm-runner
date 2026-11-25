#!/bin/sh
# Drives the README demo. Shows clean commands while running the real binary.
# Takes the model path as $1 — use the PINNED model so the GIF is reproducible:
#   MODEL=$(sh demo/fetch-model.sh)
#   asciinema rec --overwrite -c "sh demo/record.sh $MODEL" demo/demo.cast
#   agg demo/demo.cast demo/demo.gif
set -e
MODEL="${1:?usage: record.sh <model.gguf>  (get the pinned path from demo/fetch-model.sh)}"
PATH="$(cd "$(dirname "$0")/.." && pwd)/build:$PATH"   # so argv[0] is just "llm-runner"
P='\033[1;32m$\033[0m'   # green prompt

sleep 0.8
printf "${P} llm-runner -h\n"
llm-runner -h 2>&1 | head -9
sleep 2.0

printf "\n${P} llm-runner -m qwen2.5-0.5b.gguf -sys 'Answer in one short sentence.' -p 'What is the capital of France?' -n 32\n"
# stderr (model-load logs) hidden so the GIF shows just the streamed reply
llm-runner -m "$MODEL" -sys "Answer in one short sentence." -p "What is the capital of France?" -n 32 -t 0 2>/dev/null
sleep 2.2
