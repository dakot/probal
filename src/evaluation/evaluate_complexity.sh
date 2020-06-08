#!/bin/bash

dir=/home/marek/Projects/xpal/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# global settings
# shellcheck disable=SC1068
reps=$1
query_strategies="${@:2}"
data_sets="toy_nsamples=1000_nclasses=2 toy_nsamples=1000_nclasses=4 toy_nsamples=1000_nclasses=6"
budget="500"
test_ratio=1
seed=42
results_path='/home/marek/Projects/xpal/results'

# execute each configuration
i=0
for d in ${data_sets}; do
  for q in ${query_strategies}; do
    i=$(("$i"+1))
    python -u "$dir"/evaluation/experimental_setup_csv.py --data_set "$d" --results_path "$results_path" --test_ratio "$test_ratio" --query_strategy "$q" --budget "$budget" --seed "$seed" &
    if [ $(( i % $reps )) -eq 0 ]; then
      wait
    else
      printf "Continue"
    fi
  done
done
echo "$i"

