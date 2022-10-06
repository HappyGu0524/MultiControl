# MultiControl
A Distributional Lens for Multi-Aspect Controllable Text Generation


## File structure
```
├── classify
│   ├── eval.py                       \# Evaluation metric
│   ├── indep_eval.sh                 \# Evaluation shell
│   ├── logs
│   │   └── AE_final.txt              # Final Score
│   └── model                         # Evaluator checkpoints
│       ├── AGnews-checkpoint-6000    # Topic Evaluation
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       ├── Toxic-checkpoint-3000     # Toxic Evaluation. 
                                      # Noting that this is only for validation. 
                                      # We use Perspective API for final toxic evaluation
│       │   ├── config.json           
│       │   └── pytorch_model.bin
│       └── Yelp2-checkpoint-64000    # Sentiment Evaluation
│           ├── config.json
│           └── pytorch_model.bin
├── data                              # Training data
├── generate_config_final.json        # Config for Multi-Aspect Controllable Generation
├── generate_multi.py                 # Generation file
├── generate_multi.sh                 # Generation shell
├── generation_utils.py               # Intersection Searching Algorithm
├── LICENSE
├── model
│   └── AE
│       └── checkpoint-30000          # Model Checkpoint
                                      # We preserve the parameters of Encoder, Fixed Decoder and Mapping Layer all together for simplicity.
                                      # You can only keep the searched Intersection Prefixes and drop out other parts.
│           ├── config.json
│           └── pytorch_model.bin
├── model.py                          # Our model with AE structure, PrefixTuning strategy and all losses for Attribute Space.
├── README.md
├── requirements.txt                  # All based on huggingface/transformers
├── res
│   └── AE
│       └── predict_final.txt         # Final Generated Sentences
├── train_multi.py                    # Training file
└── train_multi.sh                    # Training shell
```

## Train
```
sh train_multi.sh
```
## Generate & Test
```
sh generate_multi.sh
```

## Model & Data
Checkpoint available at:
https://drive.google.com/drive/folders/14XHSG4IAGlAL9t-SYoTUKnAs5ARqHd5f?usp=sharing
