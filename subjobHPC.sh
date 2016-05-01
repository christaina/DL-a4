#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=8:00:00
#PBS -l mem=12GB
#PBS -M ceb545@nyu.edu

module purge

SRCDIR=$HOME/nameofScriptDir/
RUNDIR=$SCRATCH/nameofScriptDir/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
#cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module load torch
th main_gru.lua
