import numpy as np
from utils import mean_std_cal
from typing import Optional, Any
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from dataset import OODCustomDataset
from transforms import transformImg


class DataModule(pl.LightningDataModule):
  def __init__(self, args)->None:
    super().__init__()
    self.batch_size = args.batch_size
    self.args = args
    self.ds = None
    self.trainDS = None
    self.valDS = None
    self.train_dataset_sub = None
    self.valid_dataset_sub = None
    self.test_ds = None
    
  def _splitup(self, datasetLen:int = 0):
    '''Train and validation split up'''
    valLen = int(self.args.valSplit * datasetLen)
    trainLen =  datasetLen - valLen
    return trainLen, valLen  
     
  def setup(self, stage: Optional[str] = None):
    ''' In-built function which is called on every GPU (automatically).Called at the beginning of 
    fit (train + validate), validate, test, and predict. This is a hook when you need to build models 
    dynamically or adjust something about them. This hook is called on every process when using DDP.
    https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/core/hooks.html#DataHooks.setup'''
    
    print("self.img_h, self.img_w - ",self.args.img_h, self.args.img_w)
    trfm = transformImg(self.args.img_h, self.args.img_w)

    if not self.args.test:
      self.ds = OODCustomDataset(self.args.data_mode, self.args.rootPath, self.args.subPath, self.args.test, transform = trfm)
      print("Complete dataset length: ", len(self.ds))

      if self.args.cal_mean_std:
        mean_std_cal(self.ds, len(self.ds))

      trainLen, valLen = self._splitup(len(self.ds))
      self.trainDS, self.valDS = random_split(self.ds, [trainLen, valLen])
      print("Training dataset length: ", len(self.trainDS))
      print("Validation dataset length: ", len(self.valDS))
    else:
      self.test_ds = OODCustomDataset(self.args.data_mode, self.args.rootPath, self.args.subPath, self.args.test, transform = trfm)
      print("Complete test dataset length: ", len(self.test_ds))
    
  def train_dataloader(self):
    '''In-built function, Implement one or more PyTorch DataLoaders for training.'''

    subs = list(range(0, len(self.trainDS), self.args.subsampling_factor_train))

    self.train_dataset_sub = Subset(self.trainDS, subs)
    print("Subsampled Training Dataset: ", len(self.train_dataset_sub))
    
    train_data_loader = DataLoader(dataset= self.train_dataset_sub, batch_size= self.batch_size, 
    shuffle= self.args.shuffle, num_workers = self.args.numWorkers, 
    drop_last = self.args.dropLast)

    return train_data_loader
    
  def val_dataloader(self):
    '''In-built function, Implement one or more PyTorch DataLoaders for validation.'''

    subs = list(range(0, len(self.valDS), self.args.subsampling_factor_valid))
    self.valid_dataset_sub = Subset(self.valDS, subs)
    print("Subsampled Validation Dataset: ", len(self.valid_dataset_sub))
    return DataLoader(dataset = self.valid_dataset_sub, batch_size= self.batch_size, shuffle= self.args.val_shuffle, num_workers = self.args.numWorkers, drop_last = self.args.dropLast)

  def test_dataloader(self):
    subs = list(range(0, len(self.test_ds), self.args.subsampling_factor_test))
    self.test_dataset_sub = Subset(self.test_ds, subs)
    print("Subsampled Test Dataset: ", len(self.test_dataset_sub))
    return DataLoader(dataset = self.test_dataset_sub, batch_size = self.batch_size, num_workers = self.args.numWorkers, drop_last = self.args.dropLast)

  # @classmethod    
  # def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs): 
  #   parser = parent_parser.add_argument_group(cls.__name__)
  #   cls._add_concrete_argparse_args(parser)
  #   return parent_parser
    
  # @classmethod
  # def _add_concrete_argparse_args(cls, arg_group):
  #   pass
