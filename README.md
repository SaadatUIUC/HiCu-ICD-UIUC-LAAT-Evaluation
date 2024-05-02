# LAAT-HiCu
This branch contains the LAAT-HiCu implementation for our MLHC 2022 paper [HiCu: Leveraging Hierarchy for Curriculum Learning in Automated ICD Coding](https://arxiv.org/abs/2208.02301).

Training
----
Please refer to `run_full.sh` and `run_50.sh` for training the LAAT-HiCu models.

----
Original README from [aehrc/LAAT](https://github.com/aehrc/LAAT):
# A Label Attention Model for ICD Coding from Clinical Text <a href="https://twitter.com/intent/tweet?text=LAAT%20%28A%20Label%20Attention%20Model%20for%20ICD%20Coding%20from%20Clinical%20Text%29%20Code:&url=https%3A%2F%2Fgithub.com%2Faehrc%2FLAAT"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2F"></a>  
  
<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/aehrc/LAAT"> <a href="https://github.com/aehrc/LAAT/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/aehrc/LAAT"></a> <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/aehrc/LAAT"> <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/aehrc/LAAT"> <a href="https://github.com/aehrc/LAAT/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/aehrc/LAAT"></a> <a href="https://github.com/aehrc/LAAT/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/aehrc/LAAT"></a>    

This project provides the code for our JICAI 2020 [A Label Attention Model for ICD Coding from Clinical Text](https://arxiv.org/abs/2007.06351) paper.

The general architecture and experimental results can be found in our [paper](https://arxiv.org/abs/2007.06351):

```
  @inproceedings{ijcai2020-461-vu,
      title     = {A Label Attention Model for ICD Coding from Clinical Text},
      author    = {Vu, Thanh and Nguyen, Dat Quoc and Nguyen, Anthony},
      booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI-20}},             
      pages     = {3335--3341},
      year      = {2020},
      month     = {7},
      note      = {Main track}
      doi       = {10.24963/ijcai.2020/461},
      url       = {https://doi.org/10.24963/ijcai.2020/461},
   }
```

### Setting up the Environment Locally:

For best experience setting up the project, I recommend Anaconda, which can be downloaded from the following link: [Anaconda](https://www.anaconda.com/
)

### Requirements

Use the provided `environment.yml` to setup your local environment.

Run `conda env create -f environment.yml` to install the required libraries
Run `conda activate hicu_env`
Run `python3` and run `import nltk` and `nltk.download('punkt')` for tokenization 

### Data preparation

#### MIMIC-III-full and MIMIC-III-50 experiments
`data/mimicdata/mimic3`
The folder structure should be similar to the following

```
data
└── mimicdata/
    └── mimic3/
        ├── D_ICD_DIAGNOSES.csv
        ├── D_ICD_PROCEDURES.csv
        ├── DIAGNOSES_ICD.csv
        ├── PROCEDURES_ICD.csv
        ├── NOTEEVENTS.csv
        ├── dev_50_hadm_ids.csv
        ├── dev_full_hadm_ids.csv
        ├── test_50_hadm_ids.csv
        ├── test_full_hadm_ids.csv
        ├── train_50_hadm_ids.csv
        └── train_full_hadm_ids.csv
```

We need to load the relevant MIMIC-III files (`D_ICD_DIAGNOSES.csv`, `D_ICD_PROCEDURES.csv`, `DIAGNOSES_ICD.csv`, `PROCEDURES_ICD.csv`, and `NOTEEVENTS.csv`) into their respective tables in PostgreSQL after creating the tables.

For example:

`\COPY d_icd_diagnoses FROM '/path/to/D_ICD_DIAGNOSES.csv' DELIMITER ',' CSV HEADER;`

- The id files are from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)
- Install the MIMIC-III database with PostgreSQL following this [instruction](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/)
- Generate the train/valid/test sets using `src/util/mimiciii_data_processing.py`. (Configure the connection to PostgreSQL at Line 150)

**Note that:** The code will generate 3 files (`train.csv`, `valid.csv`, and `test.csv`) for each experiment.

### Pretrained word embeddings 
`data/embeddings`

We used `gensim` to train the embeddings (`word2vec` model) using the entire MIMIC-III discharge summary data. 

Our code also supports subword embeddings (`fastText`) which helps produce better performances (see `src/args_parser.py`).

### How to run

The problem and associated configurations are defined in `configuration/config.json`. Note that there are 3 files in each data folder (`train.csv`, `valid.csv` and `test.csv`)

There are common hyperparameters for all the models and the model-specific hyperparameters. See `src/args_parser.py` for more detail

Here is an example of using the framework on MIMIC-III dataset (50) with hierarchical join learning

```
@echo off
python -m src.run ^
    --problem_name mimic-iii_cl_50 ^
    --checkpoint_dir scratch/gobi2/wren/icd/laat/checkpoints ^
    --max_seq_length 4000 ^
    --n_epoch "1,1,1,1,50" ^
    --patience 6 ^
    --lr_scheduler_patience 2 ^
    --batch_size 8 ^
    --optimiser adamw ^
    --lr 0.0005 ^
    --dropout 0.3 ^
    --main_metric micro_f1 ^
    --save_results_on_train ^
    --embedding_mode word2vec ^
    --embedding_file data/embeddings/word2vec_sg0_100.model ^
    --joint_mode hicu ^
    --d_a 256 ^
    --metric_level -1 ^
    --loss ASL ^
    --asl_config "1,0,0.03" ^
    RNN ^
    --rnn_mode LSTM ^
    --n_layers 1 ^
    --bidirectional 1 ^
    --hidden_size 256
pause
```

