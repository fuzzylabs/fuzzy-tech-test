# Introduction

This repository is used during the Fuzzy Labs technical interview to explore different aspects of MLOps around a basic model. You'll find code to train a straightforward regression model to predict house prices, and during the interview we'll discuss and implement various MLOps capabilities on top of the model. 

# Running locally

## Pre-requisites

* Python version 3.7 or newer.
* Pip

## Setup using `pip` and `virtualenv`

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Training the model

```
python training/train.py
```

This will result in a trained model being saved to the `models/` directory.

