#!/bin/bash
#SBATCH -t 0:30:00
#SBATCH -n 6
#SBATCH -p gpu_shared_course

cp -r $HOME/DL_assignments_2019/assignment_3/templates $TMPDIR
cd $TMPDIR/templates

python a3_vae_template.py

cp results.dat $HOME/DL_assignments_2019/assignment_3/templates/results
cp model0.pt $HOME/DL_assignments_2019/assignment_3/templates
cp model5.pt $HOME/DL_assignments_2019/assignment_3/templates
cp model10.pt $HOME/DL_assignments_2019/assignment_3/templates
cp model20.pt $HOME/DL_assignments_2019/assignment_3/templates
cp model40.pt $HOME/DL_assignments_2019/assignment_3/templates