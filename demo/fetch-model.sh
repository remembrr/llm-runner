#!/bin/sh
# Pinned demo model. Ensures the exact GGUF used for the README GIF is present
# (downloading it via llama.cpp if missing) and verifies its SHA-256, so the demo
# is byte-for-byte reproducible: record.sh runs greedy (-t 0), so identical weights
# produce identical output. Prints the resolved model path on stdout; all progress
# goes to stderr. Run this before recording so the download is never in the GIF:
#
#   MODEL=$(sh demo/fetch-model.sh)
#   asciinema rec --overwrite -c "sh demo/record.sh $MODEL" demo/demo.cast
#   agg demo/demo.cast demo/demo.gif
set -e

REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
QUANT="Q4_K_M"
FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
SHA256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"

sha_of() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}
find_model() { find "$HOME/.cache/huggingface" -iname "$FILE" 2>/dev/null | head -1; }

M="$(find_model)"
if [ -z "$M" ]; then
  echo ">> downloading pinned demo model $REPO:$QUANT (~0.4 GB) ..." >&2
  llama-cli -hf "$REPO:$QUANT" -p ok -n 1 </dev/null >/dev/null 2>&1 || true
  M="$(find_model)"
fi
[ -n "$M" ] || { echo "error: could not obtain $FILE via -hf $REPO:$QUANT" >&2; exit 1; }

GOT="$(sha_of "$M")"
if [ "$GOT" != "$SHA256" ]; then
  echo "error: $FILE SHA-256 mismatch — the GIF would not be reproducible." >&2
  echo "  expected $SHA256" >&2
  echo "  got      $GOT" >&2
  exit 1
fi
echo ">> pinned demo model verified ($QUANT, sha256 ${SHA256%${SHA256#????????}}…)" >&2
echo "$M"
