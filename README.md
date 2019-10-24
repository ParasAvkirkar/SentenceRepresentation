This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Installation

This assignment is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-hw2 python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-hw2
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download glove wordvectors:
```
./download_glove.sh
```

**NOTE:** We will be using this environment to check your code, so please don't work in your default or any other python environment.

We have used type annotations in the code to help you know the method signatures. Although, we wouldn't require you to type annotate your python code, knowing types might help know what is expected. You can know more about type annotations [here](https://docs.python.org/3/library/typing.html "here").

# Data

There are two classification datasets in this assignment stored in `data/` directory:

- **IMDB Sentiment:**: It's a sample of the original dataset which has annotations on whether the imdb review is positive or negative. In our preprocessed version, positive is labeled 1 and negative is labeled 0.
 - Following are the development and test sets:
`imdb_sentiment_dev.jsonl` and `imdb_sentiment_test.jsonl`
 - There are different sized samples of the training set :
`imdb_sentiment_train_5k.jsonl`,  `imdb_sentiment_train_10k.jsonl` etc.

- **Bigram Order:**: It's a binary classification of wheter bigram is in correct order or reversed. For example, `New York` would be 1 and `York New` would be 0. It's in the same format as the imdb sentiment dataset.
 - The train, dev, test files are `bigram_order_train.jsonl`, `bigram_order_dev.jsonl` and `bigram_order_test.jsonl`.


# Code Overview

## Data Reading File

Code dealing with reading the dataset, generating and managing vocabulary, indexing the dataset to tensorizing it and loading embedding files is present in `data.py`.

## Modeling Files

The main code to build models is contained the following files:

- main_model.py
- probing_model.py
- sequence_to_vector.py

There are two kinds of models in this code: main and probing.

- The main model (`main_model.py`) is a simple classifier which can be instantiated using either DAN or GRU Sentence encoders defined (to be defined by you) in `sequence_to_vector.py`
- The probing model (`probing_model`) is built on top of a pretrained main model. It takes frozen representations from nth layer of a pretrained main model and then fits a linear model using those representations.

## Operate on Model:

The following scripts help you operate on these models: `train.py`, `predict.py`, `evaluate.py`. To understand how to use them, simply run `python train.py -h`, `python predict.py -h` etc and it will show you how to use these scripts. We will give you a high-level overview below though:

**Caveat:** The command examples below demonstrate how to use them *after filling-up* the expected code in the model code files. They won't work before that.

### Train:

The script `train.py` lets you train the `main` or `probing` models. To set it up rightly, the first argument of `train.py` must be model name: `main` or `probing`. The next two arguments need to be path to the training set and the development set. Next, based on what you model choose to train, you will be asked to pass extra configurations required by model. Try `python train.py main -h` to know about `main`'s command-line arguments.

The following command trains the `main` model using `dan` encoder:

```
python train.py main \
                  data/imdb_sentiment_train_5k.jsonl \
                  data/imdb_sentiment_dev.jsonl \
                  --seq2vec-choice dan \
                  --embedding-dim 50 \
                  --num-layers 4 \
                  --num-epochs 5 \
                  --suffix-name _dan_5k_with_emb \
                  --pretrained-embedding-file data/glove.6B.50d.txt
```

The output of this training is stored in its serialization directory, which includes all the model related files (weights, configs, vocab used, tensorboard logs). This serialization directory should be unique to each training to prevent clashes and its name can be adjusted using `suffix-name` argument. The training script automatically generates serialization directory at the path `"serialization_dirs/{model_name}_{suffix_name}"`. So in this case, the serialization directory is `serialization_dirs/main_dan_5k_with_emb`.

Similarly, to train `main` model with `gru` encoder, simply replace the occurrences of `dan` with `gru` in the above training command.


### Predict:

Once the model is trained, you can use its serialization directory and any dataset to make predictions on it. For example, the following command:

```
python predict.py serialization_dirs/main_dan_5k_with_emb \
                  data/imdb_sentiment_test.jsonl \
                  --predictions-file my_predictions.txt
```
makes prediction on `data/imdb_sentiment_test.jsonl` using trained model at `serialization_dirs/main_dan_5k_with_emb` and stores the predicted labels in `my_predictions.txt`.

In case of the predict command, you do not need to specify what model type it is. This information is stored in the serialization directory.

### Evaluate:

Once the predictions are generated you can evaluate the accuracy by passing the original dataset path and the predictions. For example:

```
python evaluate.py data/imdb_sentiment_test.jsonl my_predictions.txt
```

### Probing

Once the `main` model is trained, we can use its frozen representations at certain layer and learn a linear classifier on it by *probing* it. This essentially checks if the representation in the given layer has enough information (extractable by linear model) for the specific task.

To train a probing model, we would again use `train.py`. For example, to train a probing model at layer 3, with base `main` model stored at `serialization_dirs/main_dan_5k_with_emb` on sentiment analysis task itself, you can use the following command:

```
python train.py probing \
                  data/imdb_sentiment_train_5k.jsonl \
                  data/imdb_sentiment_dev.jsonl \
                  --base-model-dir serialization_dirs/main_dan_5k_with_emb \
                  --num-epochs 5 \
                  --layer-num 3
```

Similarly, you can also probe the same model on bigram-classification task by just replacing the datasets in the above command.

### Analysis:

There are four scripts in the code that will allow you to do analyses on the sentence representations:

1. plot_performance_against_data_size.py
2. plot_probing_performances_on_sentiment_task.py
3. plot_probing_performances_on_bigram_order_task.py
4. plot_perturbation_analysis.py

To run each of these, you would require to first train models in specific configurations. Each script has different requirements. Running these scripts would tell you these requirements are and what training/predicting commands need to be completed before generating the analysis plot. If you are half-done, it will tell you what commands are remaining yet.

Before you start with plot/analysis section make sure to clean-up your `serialization_dirs` directory, because the scripts identify what commands are to be run based on serialization directory names found in it. After a successful run, you should be able to see some corresponding plots in `plots/` directory.

Following section briefs what are the requirements of each script:

#### Performance against data size:

Compare the validation accuracy of DAN and GRU based models across 3 different training dataset sizes on sentiment analysis task.

#### Probing Performances on Sentiment Task

Probe DAN and GRU models at different layers and analyse how they perform on sentiment analysis task.

#### Probing Performances on Bigram Order Task

Probe DAN and GRU models at final layer and compare how they perform on Bigram Order classification task.

#### Perturbation Analysis

For a sample sentence "`the film performances were awesome`", change the word awesome to `worst`, `okay` and `cool`, and see how the learnt representations change at different layers for DAN and GRU models.


Lastly, you are also supposed to train DAN for more epochs and say something about training and the validation losses from the tensorboard. For this, run the script:

```
./train_dan_for_long.sh
```

and check train/validation losses on the tensorboard (`http://localhost:6006/`) after running:

```
tensorboard --logdir serialization_dirs/main_dan_5k_with_emb_for_50k
```

Note: If you run training multiple times with same name, make sure to clean-up tensorboard directory. Or else, it will have multiple plots in same chart.


# Expectations

## What to write in code:

There are only 3 placeholders in the code.

1. Deep Averaging Network: (of n layers and p dropout)
 - Please look at the details of Deep Averaging Network from this paper.
 - Make sure implement dropout correctly.

2. GRU Encoder: (of n layers):
 - Use a GRU based model. You are allowed to use tf.keras.GRU for this.
 - Make sure to use masks correctly.

3. Probing Model
 - in probing_model.py

You are only allowed to use following tf.keras APIs for implementing the above:

- `tf.keras.layers.Dense`
- `tf.keras.layers.GRU`

The code placeholders that you have to fill up are marked with :
```
# TODO(students): start
  ...
# TODO(students): end
```

You don't have to make any changes in analysis code. Just fill the above place-holders, run the analysis scripts and write you observations/explanations in the report.

We would advice you the fill the above code in the same order and use the training commands as shown in previous section to check if it's running correctly. You should be able to see the loss decreasing gradually.

## What to turn in?

1. Following two files [on Blackboard]:
 - sequence_to_vector.py
 - probing_model.py
2. zipped `plots` directory [on Blackboard]
3. zipped `serialization_dirs` directory [on Google Drive (publicly accessible)]
4. gdrive_link.txt with `wget`able link of your google drive file.

Note that `serialization_dirs` should contain all the serialization files required by the four plotting scripts. Running these scripts with your provided `sequence_to_vector.py`, `probing_model.py` and `serialization_dirs` should generate the `plots` that you submitted. As a sanity check before zipping `serialization_dirs`, make sure none of them complain that something is incomplete.

All contents submitted to blackboard should be in a zipped folder with your sbu_id, last name, first name such as xxxxxxx_last_first.zip.

### What to write in report

Details in the assignment pdf.


# Some Debugging Tips:

## Debugger

We have shifted to Tensorflow 2.0 which runs in eager mode by default. This means that your computation graph construction happens as the code the run. You can now use your debugger anywhere in your code. You can also put plain python print statements if you don't like debuggers for some reason.

If you don't know how to use a debugger, here is are some basic examples:

1. Put `import pdb; pdb.set_trace()` anywhere in your python code. When the control flow reaches there, the execution will stop and you will be provided python prompt in the terminal. You can now print and interact with variables to see what is causing problem or whether your idea to fix it will work or not.

2. If your code gives error on nth run of some routine, you can wrap it with try and except to catch it with debugger when it occurrs.
```
try:
        errorsome code
except:
        import pdb; pdb.set_trace()
```

You can ofcourse use your favorite IDE to debug, that might be better. But this bare-bone debugging would itself make your debugging much more efficient than simple print statements.

## Sanity Check

Neural networks are generally over-parameterized. They can almost always fit few examples with a very high accuracy as long as modeling and losses are setup reasonably correctly. So one sanity check is to fit a model on a training dataset of 2-3 examples. If model is struggling to decrease training loss, then something is definitely wrong.


## Nans

Nans are known to make people crazy. But most the times reason is one of the following: division by zero, exponent of very large numbers, potential infinities and log of negative numbers.
