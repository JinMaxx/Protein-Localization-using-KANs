# Protein Localization Prediction using Kolmogorov-Arnold Networks

Colab: https://colab.research.google.com/github/JinMaxx/Protein-Localization-using-KANs/blob/main/colab_notebook.ipynb

---
## Project Details

### System
Tested and developed on Linux Manjaro 25.0.7

### Local Setup
Run dependencies.sh to set up this project locally.  

If you want to run the scripts locally on your machine:  
Please set your working directory for executing the scripts to project root!  

### Directory Structure
All necessary file paths and directories can be found and set in .env
```
. (Project root)
├── source/                              Source Code
├── config.yaml                          Main configuration file
├── data/
│   ├── fasta/                           Raw input FASTA files (input Data)
│   │   ├── deeploc_our_train_set.fasta  Train set from [Light Attention]
│   │   ├── deeploc_our_val_set.fasta    Val set from [Light Attention]
│   │   ├── deeploc_test_set.fasta       Test set from [deeploc 1]
│   │   └── setHARD.fasta                Improved test set from [Light Attention]
│   ├── encodings/
│   │   └── ENCODING_MODEL/              Encoded data for model
│   │       ├── deeploc_our_train_set.h5
│   │       ├── deeploc_our_val_set.h5
│   │       ├── deeploc_test_set.h5
│   │       └── setHARD.h5
│   ├── saved_models/
│   │   └── ENCODING_MODEL/              Saved trained models
│   │       └── Model_id.pth             Pytorch model checkpoint
│   ├── figures/
│   │   └── ENCODING_MODEL/              Plots and figures
│   │       ├── figure_id.png            Image file of figure
│   │       └── figure_id.pkl            Figure as serialized python objects 
│   ├── studies/
│   │   └── ENCODING_MODEL/              Saved studies for hyper parameter tuning
│   └── logging/
│       └── ENCODING_MODEL/             
│           ├── hyper_param_metrics.tsv  metrics for models during hyper parameter optimization
│           ├── training_metrics.tsv     metrics for models during training
│           └── output.log               Logged std out for hyper_param, training and (pruning)
...
```


---

License: MIT  
This notebook and its code are made available under the MIT License.  
See [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) for details. 
 
---