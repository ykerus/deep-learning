#!/bin/bash
#SBATCH -t 0:15:00
#SBATCH -n 3
#SBATCH -p gpu_shared_course

cp -r $HOME/DL_assignments_2019/assignment_1/code $TMPDIR
cd $TMPDIR/code

python train_convnet_pytorch.py

cp results.dat $HOME/DL_assignments_2019/assignment_1/code/results
