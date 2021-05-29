# AdaLabel

Code/data for ACL'21 paper "Diversifying Dialog Generation via Adaptive Label Smoothing".

We implemented an Adaptive Label Smoothing (AdaLabel) approach that can adaptively estimate a target label distribution at each time step for different contexts.
Our method is an extension of the traditional MLE loss.
The current implementation is designed for the task of dialogue generation. 
However, our approach can be readily extended to other text generation tasks such as summarization.
Please refer to our paper for more details.

Our implementation is based on the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) project, 
therefore most behaviors of our code follow the default settings in OpenNMT-py.
Specifically, we forked from [this commit](https://github.com/OpenNMT/OpenNMT-py/tree/1bbf410a00e1d15c87fc5393b9124d531e134445) of OpenNMT-py,
and implemented our code on top of it.
This repo reserves all previous commits of OpenNMT-py and ignores all the follow-up commits.
Our changes can be viewed by comparing the [commits](https://github.com/lemon234071/AdaLabel/commit/4b9531943a4e00f1ee8a7f4b8bf3554e2b1e0f41).

Our code is tested on Ubuntu 16.04 using python 3.7.4 and pytorch 1.7.1.

## How to use

### Step 1: Setup

Install dependecies:
```bash
conda create -n adalabel python=3.7.4
conda activate adalabel
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch -n adalabel 
pip install -r requirement.txt
```
Make folders to store training and testing files:
```bash
mkdir checkpoint  # Model checkpoints will be saved here
mkdir log_dir     # The training log will be placed here
mkdir result      # The inferred results will be saved here
```

### Step 2: Preprocess the data

The data can be downloaded from this [link](https://drive.google.com/file/d/1U4M0h9tLNeCyu9JBfSgR3r5EE6IIqyNZ/view?usp=sharing).
After downloading and unziping, the DailyDialog and OpenSubtitle dataset used in our paper can be found in the `data_daily` and `data_ost` folder, respectively.
We provide a script `scripts/preprocess.sh` to preprocess the data.
```bash
bash scripts/preprocess.sh
```

Note:
- Before running `scripts/preprocess.sh`, remember to modify its firt line (i.e., the value of `DATA_DIR`) to specify the correct data folder.
- The default choice of our tokenizer is [bert-base-uncased](https://huggingface.co/bert-base-uncased)

### Step 3: Train the model

The training of our model can be performed using the following script:
```bash
bash scripts/train.sh
```
Checkpoints will be saved to the `checkpoint` folder.


Note:
- Before running `scripts/train.sh`, remember to modify its firt line (i.e., the value of `DATA_DIR`) to specify the correct data folder.
- By default, our script uses the first available GPU.
- Experiments in our paper are performed using TITAN Xp with 12GB memory.
- Once the training is completed, the training script will log out the best performing model on the validation set.

### Step 4: Inference

The inference of our model can be performed using the following script:
```bash
bash scripts/inference.sh {which GPU to use} {path to your model checkpoint}
```
Inferred outputs will be saved to the `result` folder.

Note:
- Before running `scripts/inference.sh`, remember to modify its firt line (i.e., the value of `DATA_DIR`) to specify the correct data folder.

### Step 5: Evaluation

The following script can be used to evaulate our model based on the inferred outputs obtained in Step 4:
```bash
python scripts/eval.py {path to the data folder} {path to the inferred output file}
```

## Citation

Please cite our paper if you find this repo useful :)
```
@inproceedings{wang2021adalabel,
  title={Diversifying Dialog Generation via Adaptive Label Smoothing},
  author={Wang, Yida and Zheng, Yinhe and Jiang, Yong and Huang, Minlie},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```

----

Issues and pull requests are welcomed.
