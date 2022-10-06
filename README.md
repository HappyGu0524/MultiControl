# MultiControl
A Distributional Lens for Multi-Aspect Controllable Text Generation


## File structure
```
├── classify
│   ├── eval.py
│   ├── indep_eval.sh
│   ├── logs
│   │   └── AE_final.txt
│   └── model
│       ├── AGnews-checkpoint-6000
│       ├── Toxic-checkpoint-3000
│       └── Yelp2-checkpoint-64000
├── data
├── generate_config_final.json
├── generate_multi.py
├── generate_multi.sh
├── generation_utils.py
├── LICENSE
├── model
│   └── AE
│       └── checkpoint-30000
├── model.py
├── README.md
├── requirements.txt
├── res
│   └── AE
│       └── predict_final.txt
├── train_multi.py
└── train_multi.sh
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
