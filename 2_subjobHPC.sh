#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=8:00:00
#PBS -l mem=12GB
#PBS -M ceb545@nyu.edu

module purge

SRCDIR=/scratch/ceb545/dl/DL-a4
RUNDIR=$SCRATCH/DL-a4/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
#cp -R $SRCDIR/* $RUNDIR

#cd $RUNDIR
cd $SRCDIR

module load torch
th 2_main_gru.lua
