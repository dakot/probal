# Toward Optimal Probabilistic Active Learning Using a Bayesian Approach

## Project Structure
- images: contains the visualizations of the results and utility plots
- src: Python package consisting of several sub-packages
    - base: implementation of DataSet and QueryStrategy class
    - classifier: implementation of Parzen Window Classifier (PWC)
    - evaluation: scripts for experimental setup and slurm
    - notebooks: jupyter notebooks for the investigation of the different query strategies
    - query_strategies: implementation of all query strategies
    - utils: helper functions
- data_set_ids.csv: contains the list of data sets with their IDs at [OpenML](https://www.openml.org/home)

## How to execute an experiment?
Due to the large number of experiments, we executed the experiments on a computer cluster consisting of four nodes with
about 100 CPU units. Using these nodes, we were able to execute 100 experiments simultaneously. An exemplary script how
showing how we executed the experiments on our computer cluster with SLURM is given by 
`projectpath/src/evaluation/evaluate_competitive_strategies.sh`.

Without such a computer cluster, it will  probably take several days to reproduce all results of the article. Nevertheless, one can execute the 
experiments on a local personal computer by following the upcoming steps.

1. Setup Python environment:
```bash
projectpath$ sudo apt-get install python3-pip
projectpath$ pip3 install virtualenv
projectpath$ virtualenv xpal
projectpath$ source xpal/bin/activate
projectpath$ pip3 install -r requirements.txt
```
2. Get information about the available hyperparameters (argparse) for the experiments.
```bash
projectpath$ cd src/evaluation/
projectpath/src/evaluation$ python3 experimental_setup_csv.py -h
```
3. Example experiment: To test xPAL on the dataset iris with:
    - a budget of 200 samples, 
    - a test ratio of 40%, 
    - the RBF kernel, 
    - the mean criterion as bandwidth,
    - and using the seed 1,
    
we have to execute the following commands.
```bash
projectpath/src/evaluation$ python3 experimental_setup_csv.py \
  --query_strategy xpal-0.001 \
  --data_set iris \
  --results_path ../../results \
  --test_ratio 0.4 \
  --bandwidth mean \
  --budget 200 \
  --seed 1
```
The results are saved in the directory `projectpath/results/` as a .csv-file.
The names of the possible data sets are given in the file `xpal/dat_set_ids.csv`.
The available kernels are: `rbf`, `categorical`, and `cosine`.
To create the ranking plot, there must be at least one run for each strategy on a data set. For example, you could execute the above command with the different strategies by setting the `query_strategy` parameter: `xpal-0.001, pal-1, lc, alce, qbc, zero-one-loss, random, qbc`.


## How to plot the experimental results?
Start jupyter-notebook and run the jupyter-notebook file `xpal/src/notebooks/evaluation_csv.ipynb`.
Remark: The ranking plots can only be created when we have for each dataset and each strategy the same number of 
executed experiments. 
```bash
projectpath$ source xpal/bin/activate
projectpath$ jupyter-notebook
```

## How to reproduce the utility plots?
Start jupyter-notebook and run the jupyter-notebook file `xpal/src/notebooks/visualization.ipynb`.
```bash
projectpath$ source xpal/bin/activate
projectpath$ jupyter-notebook
```

## How to reproduce the runtime plots?
Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/projectpath/complexity.ipynb`.
```bash
projectpath$ source xpal/bin/activate
projectpath$ jupyter-notebook
```