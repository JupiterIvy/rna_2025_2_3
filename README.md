## CBIS-DDSM Dataset with Efficient Net V2 M x Resnet 152

This project aims to mitigate a binary classification problem in the healthcare field, specifically regarding the prediction of benign or malignant tumors identified in mammograms.

To this end, a case study was conducted comparing two influential classification models: EfficientNetV2 and Resnet152.

The objective of the study was to compare both models and perform the necessary treatments, such as the use of a hyperparameter optimizer, Optuna, the use of stratified cross-validation, and the preprocessing of the target dataset.

In this repository, you will find the notebooks used to test both models separately.

### Preprocessing

For dataset preprocessing:

- Empty or duplicate data was checked;
- Columns were renamed to more valid variables;
- Merge between the test and training dataframes, for both the calcification and mass spreadsheets;
- Merge between the calcification and mass dataframes;
- Encoder of Malignant, Benign, and Benign with Callback attributes;
- Merge of the resulting dataframes with the dicom_data dataframe (contains direct paths to the dataset images).

## Hyperparameters

Optuna was used to generate the best hyperparameters for training.

Optuna is a Python library for hyperparameter optimization (HPO) designed to be flexible, efficient, and easy to integrate into ML workflows. Its key features are:

(1) Define-by-run API that allows building search spaces dynamically,
(2) advanced sampling mechanisms (samplers) to propose new sets of hyperparameters,
(3) pruning strategies that stop unpromising experiments early, saving time and resources.

It uses the following concepts:

Study: logical unit of optimization — a study that groups many trials (attempts) to maximize/minimize an objective function.
Trial: an execution of the objective with a specific set of hyperparameters; each trial returns a performance value (and may report intermediate values).
Trial.suggest_*: calls to the objective function (trial.suggest_float, trial.suggest_int, trial.suggest_categorical, etc.) used to define the space and request values ​​from the sampler during execution.
Sampler: component that determines how to generate new hyperparameter candidates (e.g., TPESampler, RandomSampler, CmaEsSampler, etc.).
Pruner: component that decides, based on intermediate metrics, whether a trial should be interrupted before completion (e.g., MedianPruner, SuccessiveHalvingPruner, HyperbandPruner).
Storage: backends for persistence (SQLite/MySQL/Postgres via RDBStorage, JournalStorage, etc.), necessary for saving, resuming, and parallelizing studies.

Hyperparameters studied:
unfreeze_ratio = trial.suggest_categorical("unfreeze_ratio", [0.2, 0.4, 0.6, 1.0])
batch_size = trial.suggest_categorical("batch_size", [8, 12])
optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam", "rmsprop"])
lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
dropout = trial.suggest_float("dropout", 0.0, 0.5)

## Stratified Cross-Validation

The goal of stratified cross-validation is:
To ensure that, in each partition of the cross-validation, the distribution of classes is representative of the original set. This prevents some folds from having an excessive concentration of one class, which would lead to unstable or biased evaluations.

## Training

Training was performed based on the collected hyperparameter data and using 3 folds for stratified cross-validation, due to limited computational power.

# PARTICIPANTS
- Evelyn Bessa
- Nezi Pimentel
- Sandra Valcacer
- Samira Souza
