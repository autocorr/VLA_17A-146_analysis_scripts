#!/bin/zsh

#PBS -V
#PBS -d "/lustre/aoc/users/bsvoboda/17A-146/"
#PBS -L tasks=1:lprocs=16:memory=12gb
#PBS -q batch

source /users/bsvoboda/.zshrc

python3 -m analysis_scripts.run_nestfit --run-nested


