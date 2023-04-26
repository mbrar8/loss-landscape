#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=5GB

source ../bin/activate

python3 plot_surface.py --cuda --model dense_nll --dataset MNIST --x='-1:1:35' --y='-1:1:35' --model_file MNIST/trained/dense_entropy/model_20230414_171505_10 --dir_type states --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot
