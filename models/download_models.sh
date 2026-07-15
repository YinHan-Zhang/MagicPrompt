#!/usr/bin/env bash

# ---- source repos (all on HuggingFace) -------------------------------------
T2V_REPO="Wan-AI/Wan2.2-T2V-A14B"
T2V_DIR="Wan2.2-T2V-A14B"

CTRL_REPO="Wan-AI/Wan2.2-Fun-5B-Control"
CTRL_DIR="Wan2.2-Fun-5B-Control"

WHAT="${1:-all}"

download_hf() {
    # $1 = repo id, $2 = local dir
    echo ">> Downloading $1 from HuggingFace into ./$2"
    huggingface-cli download "$1" \
        --local-dir-use-symlinks False \
        --local-dir "./$2"
}

download_t2v() {
    if [ -d "./${T2V_DIR}" ] && [ "$(ls -A "./${T2V_DIR}" 2>/dev/null)" ]; then
        echo ">> ${T2V_DIR} already exists, skipping."
        return
    fi
    download_hf "${T2V_REPO}" "${T2V_DIR}"
}

download_control() {
    if [ -d "./${CTRL_DIR}" ] && [ "$(ls -A "./${CTRL_DIR}" 2>/dev/null)" ]; then
        echo ">> ${CTRL_DIR} already exists, skipping."
        return
    fi
    download_hf "${CTRL_REPO}" "${CTRL_DIR}"
}

case "${WHAT}" in
    t2v)
        download_t2v
        ;;
    control2v)
        download_control
        ;;
    all)
        download_t2v
        download_control
        ;;
    *)
        echo "Unknown target: ${WHAT} (expected: t2v | control2v | all)"
        exit 1
        ;;
esac

echo ">> Done. Models are ready under: $(pwd)"
