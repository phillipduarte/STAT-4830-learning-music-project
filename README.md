# Music Piece Classification

**Team Members:** Phillip Duarte, Ethan Fan, Selina Zou

Below is the repository structure and instructions for reproducing main results from our project.

## Repository Structure
Ignore folders 'deprecated', 'docs'/'figures' (project instructions from Prof. Davis)
```
├── config/config.py                   # Config relevant to running data pipeline
├── data/                              # Intermediate files for initial linear/logistic regression
│   ├── ...
├── data_pipleine/                     # Main pipeline files in order (disregard misspelling)     
│   ├── data_prep.py                   # Loads from music21
│   ├── render.py                      # Renders as WAV
│   ├── embed.py                       # Embed WAV with MERT
│   └── perturb.py                     # Condensed pipeline for perturbed data 
├── deployed/                          # Pipeline refactor (main files in order)
│   ├── config/                        
│   │   ├── base.py
│   │   ├── cosine_arcface.py          # For metric learning
│   │   └── mlp.py                     # For regular MLP
│   ├── models/                        # Model classes
│   │   ├── __init__.py
│   │   ├── cosine_arcface.py
│   │   └── mlp.py
│   ├── cosine_arcface_model_spec.md   # Markdown on metric learning
│   ├── SUMMARY.md                     # Markdown on this pipeline refactor
│   ├── run.py                         # Entry point
│   ├── data.py                        # Data loading and preprocessing
│   ├── train.py                       # Training logic with flexible hyperparameters
│   ├── evaluate.py                    # Evaluation metrics
│   ├── search.py                      # Hyperparameter search
│   └── knn.py                         # KNN experiment
├── finetuning/                        # Finetuning of MERT
│   ├── config.py                      # Comparable to config.py in config folder
│   ├── finetune_caching_disk.py       # Main finetuning code
│   └── plot_finetune_log.py           # Helper file to plot results   
├── notebooks/  
│   ├── 01_data_prep_notebook.ipynb    # Load from music21 for linear regression
│   ├── 02_optimization_notebook.ipynb # Optimization for linear regression
│   ├── 02a_logistic_regression.ipynb  # Logistic regression redo of above
│   ├── w6_embeddings_logreg.ipynb     # Logistic regression with embeddings
│   ├── w7_mlp.ipynb                   # MLP with embeddings
│   ├── w10_mlp_search.ipynb           # MLP architecture search with embeddings
│   ├── w12_mlp_search.ipynb           # Repeat of above for perturbed version
├── presentation/                      # 5 previous slides submitted
│   ├── ...
├── finetune_4layers_curve.png         # Image files in reports
├── ...             
├── report.md                          # 5 previous reports submitted
├── ...
├── report5.md
├── PRIME_INTELLECT.md                 # Instructions for Prime Intellect use   
├── STAT_4830_Music_Classification.pdf # Final report PDF
├── STAT 4830 Slide Deck Final.pdf     # Final presentation slides
└── README.md                          # This file             
```

## Setup Instructions
To discuss the steps to reproduce results, follow notes here. There are several options for how to run the code which we've used such as running locally, in Colab, and with Prime Intellect. Described is the case when loading Bach chorales from music21, but note that you can also enter the pipeline using MIDI or WAV files. Following cloning the repository do:

1. **Versioning:** Python 3.11+ is required. It is advisable to use a CUDA-compatible GPU as some parts take a long time on CPU (i.e. generating perturbations, embeddings).

2. **Data Pipeline:** The files involved are in the folder 'data_pipleine'. Also, in order to render MIDI as WAV, a soundfont is required, the exact one we used can be downloaded [here](https://www.polyphone.io/en/soundfonts/instrument-sets/250-fluidr3-gm).

    a. Clean version: In order to produce audio WAV snippets of pieces separated into train/test split in the format our models require, the files that have to be ran in order are `data_prep.py` (loads as MIDI and snippets pieces from music21), `render.py` (renders MIDI as WAV), and `embed.py` (uses MERT to embed these snippets).

    b. Perturbed version: Alternatively, the file `perturb.py` (which is a combination of the above 3 files) can be run which will do the exact same thing but create perturbed versions of each snippet as specified (see the function 'perturb_snippet'). The result of both of these pipelines is a set of WAV files with associated `manifest.csv` with info on each and whether they are in train/test.

    In both cases, some installs to run before running the code are (Colab version)
    ```
    !pip install numpy soundfile librosa music21 transformers matplotlib
    ```
    The following is needed for `render.py`
    ```
    !apt-get install -y fluidsynth libsndfile1
    ```
    
    The specific directory structure (of all the data) should be like this. In our work we used a Google Drive folder to keep all this. It is pretty much analogous for the perturbed data, except the snippets are created after rendered as WAV rather than before. Ensure that 
    ```
    ├── snippets/                          # MIDI snippets
    │   ├── manifest.csv                   # Table about the snippets contained herein
    │   ├── bach__bwv1.6__s0000.mid
    │   └── ...
    ├── audio/                             # WAV snippets
    │   ├── bach__bwv1.6__s0000.wav
    │   └── ...
    ├── embeddings/                        # Embeddings as numpy arrays etc.
    │   ├── embeddings_train.npy
    │   ├── embeddings_test.npy
    │   ├── labels_train.npy               
    │   ├── labels_test.npy
    │   └── ...                            # Visualizations generated stored here too
    ├── finetune/                          # Populated later in finetuning
    │   ├── finetune_log.csv               # Log of top-1/top-5 per epoch
    │   ├── finetune_best.pt               # Best model weights saved
    │   └── ...                            # Cached frozen layers, etc.
    └── config.py                          # Required for training
    ```

3. **Running things locally or in Colab:** In terms of next steps in training classification models on these embeddings of music piece snippets that have just been generated, the first way is to run things locally or in Google Colab. The latter is recommended since GPUs such as T4, A100, etc. may be used. The important thing to note here is that the file `config.py` (in the config folder) should be in the root folder from which things are being run, as the below files reference it to import config info.
- The relevant files are the Python notebooks namely `w6_embeddings_logreg.ipynb` (logistic regression), `w7_mlp.ipynb` (basic MLP), `w10_mlp_search.ipynb` and its counterpart `w12_mlp_search_ipynb` (search over MLP architectures). These directly are ipynb so can be run cell by cell in Colab.
- For finetuning there is `finetune_caching_disk.py` which is a Python file, but can also be ran in Colab by running a cell like `!python /content/drive/MyDrive/stat-4830/finetune_caching_disk.py`. See the exact file for where results may be saved.
- If running the 'perturb' version of things, make sure that the file structure matches that which is required or change variables to make things match
- Also note, if it is of interest, the earliest work we did with logistic regression on extracted musical features (key, tempo, etc.) can all be easily run locally as those do not take too long.

4. **Running things in Prime Intellect:** To utilize Prime Intellect's compute to perform some of the more computationally expensive work, there are also a few options (see PRIME_INTELLECT.md for more detailed instructions). In both cases one should do the same 'pip install' (such as in step 2b as needed)
- The first option is to copy all necessary files from local into the GPU instance. We used `rclone` for downloading from Google Drive and then something like `gpu: scp -r <local_path> ubuntu@<remote_path>:~` for the copying. The file can then be run with `python3 <path_to_file>`. Afterwards, files generated that are in the remote machine can be copied back into local in a similar manner.
- The second option is using `uv` which doesn't require the manual copying. This allows for better reproduciblity and the specific directions are in `deployed/PRIME_INTELLECT.md`

## Demo
https://www.youtube.com/watch?v=h-eBnwdgSR0
