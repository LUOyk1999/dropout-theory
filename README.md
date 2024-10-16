## Python environment setup with Conda

Tested with Python 3.7, PyTorch 1.12.1, and PyTorch Geometric 2.3.1, dgl 1.0.2.
```bash
pip install pandas
pip install scikit_learn
pip install numpy
pip install scipy
pip install einops
pip install ogb
pip install pyyaml
pip install googledrivedownloader
pip install networkx
pip install gdown
pip install matplotlib
```

Running MPNNs on meidum graphs:
```bash
sh run_bn.sh 0 > mpnn_bn.txt 2>&1 &
```

The results of the hyperparameter tuning are in the "results" folder, where you can find the optimal hyperparameters for each dataset.

Running MPNNs on ogbn-products (mini-batch training):
```bash
sh products.sh
```