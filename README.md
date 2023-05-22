# MultiControl & PriorControl
**MultiControl**: [A Distributional Lens for Multi-Aspect Controllable Text Generation](https://arxiv.org/pdf/2210.02889.pdf) *EMNLP 2022 Oral*

**PriotControl**: [Controllable Text Generation via Probability Density Estimation in the Latent Space](https://arxiv.org/pdf/2212.08307.pdf) *ACL 2023*


## File structure
```
├── LICENSE
├── README.md
├── classify
│   ├── eval.py                       # Evaluation metric
│   ├── indep_eval.sh                 # Evaluation shell
│   ├── logs
│   │   └── AE_final.txt              # Final Score for MultiControl
│   └── model                         # Evaluator checkpoints
│       ├── AGnews-checkpoint-6000    # Topic Evaluation
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       ├── Toxic-checkpoint-3000     # Toxic Evaluation
                                      # Noting that this is only for validation. 
                                      # We use Perspective API for final toxic evaluation
│       │   ├── config.json           
│       │   └── pytorch_model.bin
│       └── Yelp2-checkpoint-64000    # Sentiment Evaluation
│           ├── config.json
│           └── pytorch_model.bin
├── data                              # Training data
├── model
│   └── multicontrol                  # Model Folder for MultiControl
│   │   └── checkpoint-30000          # Model Checkpoint
    │                                 # We preserve the parameters of Encoder, Fixed Decoder, and Mapping Layer all together for simplicity.
    │                                 # You can only keep the searched Intersection Prefixes and drop out other parts.
│   │       └── pytorch_model.bin
│   └── priorcontrol                  # Model Folder for PriorControl
│       └── All_notricks_checkpoint-300000
                                      # We preserve the parameters of Encoder, Fixed Decoder, Mapping Layer, and Normalizing Flows all together for simplicity.
                                      # You need to keep the Prior Heads and Mapping Layer before drop out other parts.
│           └── pytorch_model.bin
├── multicontrol                      # Code and Script Folder for MultiControl
│   ├── generate_config_final.json    # Config for Multi-Aspect Controllable Generation
│   ├── generate_multi.py             # Generation file
│   ├── generate_multi.sh             # Generation shell
│   ├── generation_utils.py           # Intersection Searching Algorithm
│   ├── model.py                      # Our model with AE structure, PrefixTuning strategy and all losses for Attribute Space
│   ├── requirements.txt              # All based on huggingface/transformers
│   ├── train_multi.py                # Training file
│   └── train_multi.sh                # Training shell
├── priorcontrol                      # Code and Script Folder for PriorControl
│   ├── logs                          # Folder for Final Scores
│   ├── combine_eval.sh               # Evaluation Script for Multi-Attribute Control
│   ├── generate_combine_optim.py     # Code for Multi-Attribute Control via Optimization
│   ├── generate_combine.py           # Code for Multi-Attribute Control via Interpolation
│   ├── generate_combine.sh           # Script for Multi-Attribute Control
│   ├── generate_config_combine_optim.json  # Config for Multi-Aspect Control via Optimization
│   ├── generate_config_combine.json  # Config for Multi-Aspect Control
│   ├── generate_prior.py             # Code for Single-Attribute Control
│   ├── generate_prior.sh             # Script for Single-Attribute Control
│   ├── latentops_modules.py          # Code Modified from LatentOps which provides ODE for Optimization
│   ├── model.py                      # Our model with AE structure, PrefixTuning strategy, Normalizing Flows, and all losses for Attribute and Prior Space
│   ├── requirements.txt              # Framework: huggingface/transformers, Normalizing Flows: FrEIA, ODE Optimization: torchdiffeq
│   ├── single_eval.sh                # Evaluation Script for Single-Attribute Control
│   ├── train_prior_only.py           # Training file for Normalizing Flows with fixed AE
│   └── train_prior_only.sh           # Training Script
└── res
    └── multicontrol
    │   └── predict_final.txt         # Final Generated Sentences for MultiControl
    └── priorcontrol                  # Final Generated Sentences for PriorControl
        ├── generate_combination_optim.txt         
        ├── generate_combination_optimcons.txt
        ├── generate_combination.txt
        ├── generate_prior_extend.txt
        └── generate_prior.txt

```
## MultiControl
### Train
```
sh train_multi.sh
```
### Generate & Test
```
sh generate_multi.sh
```

## PriorControl
### Train
```
sh train_prior_only.sh
```
### Generate & Test
```
#Single-Attribute Control

sh generate_prior.sh
sh single_eval.sh

#Multi-Attribute Control

sh generate_combine.sh
sh combine_eval.sh
```

## Model & Data

Checkpoint available at:
https://drive.google.com/drive/folders/14XHSG4IAGlAL9t-SYoTUKnAs5ARqHd5f?usp=sharing
