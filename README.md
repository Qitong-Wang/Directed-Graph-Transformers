# Directed Graph Transformer 

The general framework is developed by Edge-augmented Graph Transformer (PyTorch) https://github.com/shamim-hussain/egt_pytorch

Datasets locate at ./raw_data directory.  Here is the link for more datasets: https://drive.google.com/file/d/1vRL-Jz-RZmA7x9qK-j5XaamvDGxOCZR4/view?usp=sharing

Configurations locate at ./configs directory.  Example commands of training: 
```
python train_graph_classification.py ./configs/Flow2/Flow.yaml  
python train_graph_classification.py ./configs/Flow3/Flow.yaml  
python train_graph_classification.py ./configs/Flow6/Flow.yaml  
python train_graph_classification.py ./configs/Perturbed_twitter3/Twitter.yaml 
python train_graph_classification.py ./configs/Perturbed_twitter5/Twitter.yaml 
python train_graph_classification.py ./configs/MALNETSub/MALNETSub.yaml  
python train_graph_classification.py ./configs/MNIST/MNIST.yaml 
python train_graph_classification.py ./configs/CIFAR/CIFAR.yaml 
```



