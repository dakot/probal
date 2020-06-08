#!/bin/bash
#
#SBATCH --job-name=xpal-evaluation
#SBATCH --output=log_slurm/slurm-xpal-evaluation.log
#SBATCH --get-user-env

#SBATCH --partition=run
#SBATCH --spread-job

#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1

source python-virtual-environments/machine_learning/bin/activate

dir=projects/xpal/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# available data sets:
# breast-cancer-wisconsin blood-transfusion pima-indians-diabetes ionosphere sonar biodegradation vehicle ecoli glass
# vertebra-column user-knowledge kc2 parkinsons banknote seeds prnn-craps chscase-vine wine iris

# global settings
query_strategies="$@"
data_sets="breast-cancer-wisconsin blood-transfusion pima-indians-diabetes ionosphere sonar
           biodegradation vehicle ecoli glass vertebra-column user-knowledge kc2 parkinsons
           banknote seeds prnn-craps chscase-vine wine iris balance-scale seismic-bumps steel-plates-fault"
budget=200
test_ratio=0.4
kernel='rbf'
bandwidth='mean'
results_path='/results_xpal'

# execute each configuration
i=0
for d in ${data_sets}; do
  for q in ${query_strategies}; do
    for s in {1..100}; do
      i=$(("$i"+1))
      srun -n1 -N1 --output='log/'$d'_'$q'_'$s'.log'  python -u "$dir"/evaluation/experimental_setup_csv.py --data_set "$d" --results_path "$results_path" --test_ratio "$test_ratio" --query_strategy "$q" --budget "$budget" --kernel "$kernel" --bandwidth "$bandwidth" --seed "$s" &
    done
    wait
  done
done
echo "$i"
wait

