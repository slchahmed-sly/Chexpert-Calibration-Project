#!/bin/bash
#SBATCH --job-name=E14_effv2s_320
#SBATCH --partition=gpu
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/chexpert_project/outputs/logs/train_E14_effv2s_320_%j.out
#SBATCH --error=/home/%u/chexpert_project/outputs/logs/train_E14_effv2s_320_%j.err

export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

source ~/.bashrc
conda activate chexpert
~/.conda/envs/chexpert/bin/python ~/chexpert_project/scripts/train_E14_effv2s_320.py
