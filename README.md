# Deep Learning Discriminative and Generative Models


The objective of this work is to develop deep learning discriminative and generative models, applied to the
context of “deep fakes”. The discriminative models will be designed to classify images as “real” vs. “fake”,
whereas the generative models will be trained to produce new “fake” examples.


### Project Structure and Architecture
```bash
├── README.md
├── configs                                         # Different configurations for model training experiments  
│   ├── baseline.json
│   ├── baseline_vae.json
│   ├── efficientnet.json
│   ├── kl_annealing_vae_baseline.json
│   └── latent_capacity_vae.json
├── models                                         # Models architecture definition
│   └── variational_autoencoder.py
├── notebooks                                      # Notebook for visualization of different pipeline stages
│   └── dataset.ipynb
├── requirements.txt                               # Dependencies
├── src                                            # Source code to build the pipeline
│   ├── build_metadata.py                          # Metadata creation - builds a file to guide data usage through training / evaluation
│   ├── classifier.py                              # THIS SHOULD BE MOVED TO models/ !!!
│   ├── dataset.py                                 # Dataset Class creation (from guidelines in Metadata)
│   ├── preprocessing.py                           # Preprocessing techniques for different models and different data splits
│   ├── setup.py                                   # Model training setup - gets the available device, builds the data loaders and applies preprocessing
│   ├── train.py                                   # THIS SHOULD BE MOVED TO training/ !!!
│   └── utils.py                                   # Helper functions
└── training                                       # Model training logic
    └── train_vae.py
```




### Authors: 
- Rodrigo Taveira
- Rodrigo Batista
- Sara Táboas
