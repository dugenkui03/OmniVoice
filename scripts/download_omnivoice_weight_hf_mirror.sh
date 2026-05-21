#!/usr/bin/env bash

# Download the missing OmniVoice main model weight through hf-mirror.
# Usage:
#   bash scripts/download_omnivoice_weight_hf_mirror.sh
#
# Optional environment variables:
#   MODEL_DIR=/path/to/OmniVoice
#   HF_ENDPOINT=https://hf-mirror.com

set -Eeuo pipefail

REPO_ID="k2-fsa/OmniVoice"
FILENAME="model.safetensors"
MODEL_DIR="${MODEL_DIR:-/Users/dugenkui/workspace/OmniVoice/.models/OmniVoice}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MIN_BYTES=2400000000

# Print a timestamped progress message so long downloads are easier to follow.
log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# Return the size of a local file in bytes, or 0 when it does not exist yet.
file_size_bytes() {
  local path="$1"

  if [[ -f "$path" ]]; then
    wc -c < "$path" | tr -d ' '
  else
    printf '0'
  fi
}

# Check that uv is available before attempting to run the HuggingFace CLI.
require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    log "ERROR: uv is not installed or not on PATH."
    exit 1
  fi
}

# Disable common proxy environment variables for this process only.
# This helps ensure the request uses your current non-VPN network path.
clear_proxy_env() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset http_proxy https_proxy all_proxy no_proxy
}

# Download the main safetensors file from the configured HuggingFace endpoint.
download_main_weight() {
  mkdir -p "$MODEL_DIR"

  log "Repo: $REPO_ID"
  log "File: $FILENAME"
  log "Endpoint: $HF_ENDPOINT"
  log "Target dir: $MODEL_DIR"

  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    uv run hf download "$REPO_ID" "$FILENAME" --local-dir "$MODEL_DIR"
}

# Validate the downloaded weight by checking that it is close to the expected size.
verify_main_weight() {
  local target_path="$MODEL_DIR/$FILENAME"
  local size

  size="$(file_size_bytes "$target_path")"
  if (( size < MIN_BYTES )); then
    log "ERROR: $target_path is only $size bytes; expected at least $MIN_BYTES bytes."
    exit 1
  fi

  log "OK: downloaded $target_path ($size bytes)."
}

require_uv
clear_proxy_env
download_main_weight
verify_main_weight

