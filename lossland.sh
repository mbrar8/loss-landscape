#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=5GB

source ../loss_env/bin/activate

python3 plot_surface.py --cuda --model dense_entropy --dataset MNIST --x='-1:1:25' --y='-1:1:25' --model_file MNIST/trained/dense_entropy/model_20230414_171505_10 --dir_type states --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot
