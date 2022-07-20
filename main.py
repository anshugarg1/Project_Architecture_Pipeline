import os
import umap
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import pandas as pd
# import seaborn as sns
from sklearn.manifold import TSNE
from sbo import soft_brownian_offset, gaussian_hyperspheric_offset

import torch
import mlflow
import pytorch_lightning as pl
from torchvision import datasets as datasets
from pytorch_lightning.loggers import MLFlowLogger 

from helper.helper import _load_ls, _generate_sbo, _save_ls, _common_percentage_btw_ls_oddls, _generate_umap, _vis_umap
# import optuna
# from optuna.integration import PyTorchLightningPruningCallback

from train import testmodel
from utils import print_auto_logged_info
from dataloader_module import DataModule

def argument_parse():
  parser = ArgumentParser()
  parser.add_argument('--log_dir', type = str, default = 'log_tester', help = "Directory to store the logs.")
  parser.add_argument('--log_folder_name', type = str, default = 'model_1_output', help = "Log folder name.")
  parser.add_argument('--lr', type = float, default= 1e-4, help = "Learning Rate.")  #passed from cmd line
  parser.add_argument('--weight_decay', type = float, default= 1e-5, help = "Learning Rate.")  #passed from cmd line
  parser.add_argument('--batch_size', type = int, default= 32, help = "DataModule args tester")
  parser.add_argument('--optimiser', default = 'AdamW', help = "mention the optimiser.") #passed from cmd line
  parser.add_argument('--epoch', type = int, default= 50, help = "Number of epochs.")  #passed from cmd line

  #train.py arguments -
  parser.add_argument('--saveImgAfter', type = int, default= 1, help = "After how many iterations should the image be saved?")
  
  #dataloader_module.py arguments -
  parser.add_argument('--rootPath', type = str, default= "/storage/Oxford_Robotcar/", help = "DataModule args tester")
  parser.add_argument('--subPath', type = str, default= "stereo/centre_colored/", help = "DataModule args tester")
  parser.add_argument('--valSplit', type = float, default= 0.2, help = "DataModule args tester")
  parser.add_argument('--seed', type = int, default= 42, help = "DataModule args tester")
  parser.add_argument('--shuffle', type = bool, default= True, help = "DataModule args tester")
  parser.add_argument('--val_shuffle', type = bool, default= False, help = "DataModule args tester")
  parser.add_argument('--numWorkers', type = int, default= 8, help = "DataModule args tester")
  parser.add_argument('--dropLast', type = bool, default= True, help = "DataModule args tester")
  parser.add_argument('--subsampling_factor_train', type = int, default= 8, help = "DataModule args tester")
  parser.add_argument('--subsampling_factor_valid', type = int, default= 8, help = "DataModule args tester")
  parser.add_argument('--subsampling_factor_test', type = int, default= 1, help = "DataModule args tester")
  parser.add_argument('--img_h', type = int, default= 256, help = "Image height")
  parser.add_argument('--img_w', type = int, default= 512, help = "Image width")
  parser.add_argument('--cal_mean_std', type = bool, default= False, help = "Calculate Mean and std?")
  
  #joint VAE parameters
  parser.add_argument('--in_channels', type = int, default= 3, help = "in_channels.")
  parser.add_argument('--latent_dim', type = int, default= 50, help = "latent_dim.")
  parser.add_argument('--categorical_dim', type = int, default= 40, help = "categorical dim.")
  parser.add_argument('--latent_min_capacity', type = float, default= 0., help = "latent min capacity.") #1
  parser.add_argument('--latent_max_capacity', type = float, default= 25., help = "latent max capacity.") #25
  parser.add_argument('--gamma', type = int, default= 30., help = "gamma.") #1000
  parser.add_argument('--latent_num_iter', type = int, default= 25000, help = "latent num iter.")
  parser.add_argument('--categorical_min_capacity', type = float, default= 0., help = "categorical min capacity.") #1
  parser.add_argument('--categorical_max_capacity', type = float, default= 25., help = "categorical max capacity.") #20
  parser.add_argument('--categorical_num_iter', type = int, default= 25000, help = "categorical num iter.")
  parser.add_argument('--temperature', type = float, default= 0.5, help = "Temperature.")
  parser.add_argument('--anneal_rate', type = float, default= 3e-5, help = "anneal_rate.") #3.0e-05
  parser.add_argument('--anneal_interval', type = int, default= 100, help = "Anneal Interval.") #10
  parser.add_argument('--alpha', type = float, default= 30, help = "alpha.") #-1
  parser.add_argument('--residual_block', type = bool, default= True, help = "Training?") #True
  parser.add_argument('--number_of_residual_blocks', type = int, default= 6, help = "Anneal Interval.") #3
  parser.add_argument('--variational_beta', type = float, default= 10.5, help = "variational Beta.")
  parser.add_argument('--training', type = bool, default= True, help = "Training?")
  parser.add_argument('--categorical_gamma', type = float, default= 30., help = "latent max capacity.")

  #sbo params
  parser.add_argument('--d_min', type = float, default= 100 , help = "variational Beta.") #0.35, 2
  parser.add_argument('--d_off', type = float, default= 100, help = "variational Beta.") #0.24, 2
  parser.add_argument('--softness', type = int, default= 1, help = "variational Beta.") #0
  
  parser.add_argument('--data_mode', type = str, default= "test_snow", help = "")
  parser.add_argument('--test', type = bool, default= True, help = "Is pipeline in testing mode?")
  parser.add_argument('--model_checkpoint', type = str, default= "/home/garg/Code/Projects/odd-generation/odd_generation/script/mlruns/7/1bb7452472164606892d44f53402c7db/checkpoint/model.pt", help = "")

  #UMAP params
  parser.add_argument('--plot_umap', type = bool, default= False, help = "")
  parser.add_argument('--neighbors', type = int, default= 20, help = "") #15
  parser.add_argument('--smaller_ood_root', type = str, default= "/home/garg/Code/Projects/odd-generation/odd_generation/test_ood/trainSun_full_smalldmindoff_22010", help = "") #

  parser.add_argument('--ls_path', type = str, default= "/home/garg/Code/Projects/odd-generation/odd_generation/script/mlruns/7/1bb7452472164606892d44f53402c7db/latent_space/49.npy", help = "") #euclidean
  parser.add_argument('--ood_points', type = int, default= 1000, help = "Points to generate OOD") #15
  
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = argument_parse()
  # create dictionary out of args and then mlflow.log_params() function. 
  dataModule = DataModule(args)
  model = testmodel(args)

  #load pretrained model weights in testing phase.
  if args.test:
    model_checkpoint = torch.load(args.model_checkpoint) 
    updated_model_checkpoint = {}
    for key in model_checkpoint.keys():
      updated_model_checkpoint['AEModel.'+key] = model_checkpoint[key]
    model.load_state_dict(updated_model_checkpoint)
    model.eval()

  # experiment_id = mlflow.create_experiment("JointVAE odd generation") 
  exp_name = "JointVAE odd generation"
  exp = mlflow.get_experiment_by_name(exp_name)
  print("Experiment_id: {}".format(exp.experiment_id))
  print("Artifact Location: {}".format(exp.artifact_location))
  print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
  
  trainer = pl.Trainer(logger = False, gpus = 1, max_epochs = args.epoch, track_grad_norm = 2) #logger = logger, 
  #log parameters
  with mlflow.start_run(experiment_id = exp.experiment_id, run_name = "Delete" , nested = True) as run:   #experiment_id=experiment_id, 
    print("mlflow.get_artifact_uri() -- ", mlflow.get_artifact_uri())
    mlflow.log_params(vars(args))
    if not args.test:
      trainer.fit(model, dataModule)
    else: 
      trainer.test(model, dataModule)

  print("mlflow.get_artifact_uri() -- ", mlflow.get_artifact_uri())
  print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
