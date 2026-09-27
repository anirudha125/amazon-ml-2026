#!/usr/bin/env bash
# One-time environment setup on Lightning.ai (Linux). Run from the repo root:  bash tools/setup_lightning.sh
set -euo pipefail
echo "== machine =="; nproc; free -g | head -2; df -h . | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv || true
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# GPU stack (adjust the CUDA wheel index to the machine if needed)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 || pip install torch
pip install transformers==5.0.0 sentence-transformers==5.4.1
python - <<'EOF'
import numpy, lightgbm, rapidfuzz, numba, sklearn, torch
print("numpy", numpy.__version__, "lightgbm", lightgbm.__version__, "rapidfuzz", rapidfuzz.__version__,
      "numba", numba.__version__, "sklearn", sklearn.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
# unpack the production reranker
mkdir -p experiments/E017_embed/reranker_e5small_full
python3 -c "import zipfile; zipfile.ZipFile('experiments/E017_embed/reranker_e5small_full.zip').extractall('experiments/E017_embed/reranker_e5small_full')"
echo "setup done -> now run: bash tools/verify_parity.sh"
