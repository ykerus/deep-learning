#!/bin/bash
#SBATCH -t 0:30:00
#SBATCH -n 6
#SBATCH -p gpu_shared_course

cp -r $HOME/DL_assignments_2019/assignment_2/part2 $TMPDIR
cd $TMPDIR/part2

python train.py --txt_file="fairytales.txt" --train_steps=50000 --out_file="FAIRY2.dat" --print_every=100 --sample_every=250 --out_seq=150

cp FAIRY2.dat $HOME/DL_assignments_2019/assignment_2/part2/results
cp FAIRY2.txt $HOME/DL_assignments_2019/assignment_2/part2/results
