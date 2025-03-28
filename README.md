This is the implementation of LResNet as residual connection in GNN architectures, where the base model is HyboNet from [Fully Hyperbolic Neural Networks].(https://arxiv.org/abs/2105.14686).
# Environment Configuration  
Install the required packages by running:  
```bash
pip install -r requirements.txt
```  

# Training  
Execute the training script by running:  
```bash
python train.py  --model "HyboNet" --num_layers 2 --dropout 0.0 --bias 1 --use_att 1 --local_agg 0 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1  --normalize_adj 1 --split_seed 1234 --manifold "Lorentz" --act None --dim 128 --task "lp" --c 1.0 --cuda -1 --n_heads 8 --alpha 0.2 --pretrained_embeddings None  --save 0 --patience 200 --min_epochs 200  --scales 1 2 3 4 
```  

# Available Components  
- **Encoders**: All supported encoders are defined in `models/encoders.py`.  
- **Manifolds**: Available manifolds are located in the `manifolds/` directory.  
- **Multi-scale Wavelet Transform**: Implemented in `models/models.py`.  
