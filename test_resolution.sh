#!/usr/bin/env bash
# resolution-matrix test: mirrors run_quant.sh logic, prints resolved SRC/OUT
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set -a
# shellcheck disable=SC1091
[ -f "$SCRIPT_DIR/.env" ] && . "$SCRIPT_DIR/.env"
set +a
MODELS_DIR="${MODELS_DIR:-./models}"
MODELS_DIR="${MODELS_DIR//\\//}" # G:\models -> G:/models
case "$MODELS_DIR" in
  [A-Za-z]:/*) _drv="${MODELS_DIR:0:1}"; MODELS_DIR="/mnt/${_drv,}${MODELS_DIR:2}" ;;
esac

resolve() {
  local src="$1"
  src="${src//\\//}"
  case "$src" in
    /*|[A-Za-z]:/*|*/*) ;;
    *) src="$MODELS_DIR/$src" ;;
  esac
  case "$src" in
    [A-Za-z]:/*) _drv="${src:0:1}"; src="/mnt/${_drv,}${src:2}" ;;
  esac
  local out="$2"
  if [ -z "$out" ]; then
    case "$src" in
      */*) out="$MODELS_DIR/$(basename "$src" | sed 's/-BF16$//' | sed 's/-bf16$//')-W4A16" ;;
      *) out="$MODELS_DIR/$(basename "$src")-W4A16" ;;
    esac
  fi
  out="${out//\\//}"
  case "$out" in
    [A-Za-z]:/*) _drv="${out:0:1}"; out="/mnt/${_drv,}${out:2}" ;;
  esac
  echo "  SRC=$src"
  echo "  OUT=$out"
}

echo "T1 bare name (positional):";        resolve "Swift-Qwen3.8-27B-Uncensored-BF16" ""
echo "T2 explicit src + out (win paths):"; resolve "G:/models/Swift-Qwen3.8-27B-Uncensored-BF16" "G:/models/My-Custom-Name"
echo "T3 QUANT_MODEL env (HF repo id):";   resolve "hotdogs/Qwen3.8-27B-thinkingcap-abliterated" ""
echo "T4 legacy name (no -BF16 suffix):";  resolve "Qwen3.8-27B-Uncensored-BF16" ""
echo "T5 HF repo id via env:";             resolve "org/some-model-bf16" ""
echo "T6 path with backslashes:";          resolve 'G:\models\Swift-Qwen3.8-27B-Uncensored-BF16' ""
echo "T7 win abs path, forward slashes:";  resolve "G:/models/Swift-Qwen3.8-27B-Uncensored-BF16" ""