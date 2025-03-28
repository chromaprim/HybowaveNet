This is the implementation of LResNet as residual connection in GNN architectures, where the base model is HyboNet from Fully Hyperbolic Neural Networks.
# Environment Configuration  
Install the required packages by running:  
```bash
pip install -r requirements.txt
```  

# Training  
Execute the training script by running:  
```bash
python train.py
```  

# Available Components  
- **Encoders**: All supported encoders are defined in `models/encoders.py`.  
- **Manifolds**: Available manifolds are located in the `manifolds/` directory.  
- **Multi-scale Wavelet Transform**: Implemented in `models/models.py`.  
