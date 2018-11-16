#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=5   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2666M   # memory per CPU core

cd /fslhome/iclee141/
source pytorch_env/bin/activate

cd cs501r/gan/

if [ "$1" != "" ]; then
    echo "python gan_trainer.py" $1
    python gan_trainer.py $1 cont
else
    echo "python gan_trainer.py sample_config.json"
    python gan_trainer.py sample_config.json
fi
