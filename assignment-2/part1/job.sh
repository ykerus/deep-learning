#!/bin/bash
#SBATCH -t 0:03:00
#SBATCH -n 4
#SBATCH -p gpu_shared_course

cp -r $HOME/DL_assignments_2019/assignment_2/part1 $TMPDIR
cd $TMPDIR/part1

python train.py --train_steps=300 --learning_rate=0.001 --input_length=33 --out_file="TEST1.dat" --eval_freq=5 --model_type="LSTM"
python train.py --train_steps=300 --learning_rate=0.001 --input_length=33 --out_file="TEST2.dat" --eval_freq=5 --model_type="RNN"

cp TEST1.dat $HOME/DL_assignments_2019/assignment_2/part1/results
cp TEST2.dat $HOME/DL_assignments_2019/assignment_2/part1/results