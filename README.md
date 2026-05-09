# FedCIGAR
 This repository is the official implementation of "[FedCIGAR: A Personalized Reconstruction Approach for Federated Graph-level Anomaly Detection]"


# Setup
```js/java/c#/text
conda create -n FedCIGAR python=3.8.20
conda activate FedCIGAR
pip install torch==1.12.1 --index-url https://download.pytorch.org/whl/cu116

export PYG_WHL=https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-scatter==2.0.9 -f $PYG_WHL
pip install torch-sparse==0.6.14 -f $PYG_WHL
pip install torch-cluster==1.6.0 -f $PYG_WHL
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric==2.1.0.post

pip install scipy
pip install numpy
pip install networkx
pip install pandas
pip install dtaidistance
```
