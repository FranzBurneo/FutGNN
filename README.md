# FutGNN — Red de pases (setup rápido)

## Requisitos
- Anaconda/Miniconda (Windows)
- Python 3.10
- Git (opcional), VS Code/Jupyter

## Instalación
```bash
# Clonar
git clone
cd FutGNN

# Entorno
conda create -n futgnn python=3.10 -y
conda activate futgnn

# PyTorch (CPU)
conda install -c pytorch pytorch torchvision torchaudio cpuonly -y
# (GPU opcional, CUDA 12.1):
# conda install -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1 -c nvidia -y

# Paquetes base
conda install -c conda-forge pandas networkx matplotlib scipy ipykernel jupyterlab tqdm fsspec -y

# PyTorch Geometric (ajusta a tu versión de torch si cambia)
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

# Kernel de Jupyter
python -m ipykernel install --user --name futgnn --display-name "Python (futgnn)"

#
python scripts/train_lp_cli.py --glob "data/raw/*.csv" --epochs 3 --batch 16 --step 1 --neg 5 --lr 0.001 --hidden 64 --layers 2 --out 64 --device cpu --val_ratio 0.33

python scripts/train_lp_cli.py --glob "data/raw/*.csv" --epochs 30 --batch 16 --step 1 --neg 5 --lr 0.001 --hidden 64 --layers 2 --out 64 --device cpu
