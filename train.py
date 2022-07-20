# %%
import os 
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch
import mlflow
# import pytorch_ssim
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.utils import save_image

from models.model import ConvolutionalVAE
from models.joint_vae import JointVAE

class testmodel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.AEModel = JointVAE(args)
        # self.AEModel = ConvolutionalVAE()

    def forward(self, x):
        '''In-built function, called automatically.'''
        return self.AEModel(x)
    
    def training_step(self, batch, batch_idx):
        '''In-built function, replaces the training loop.'''
        x, y = batch        
        out = self(x)

        recons, mu, log_var, latent_space, odd_recons, q, mod_recons = out[0], out[1], out[2], out[3], out[4], out[5], out[6]

        loss = self.AEModel.loss_function(recons, x, mu, log_var, q, batch_idx, self.current_epoch)
        run_id = mlflow.active_run()
        print("run_id train - ", run_id)
        mlflow.log_metric("Train_Loss", float(loss['loss']))
        mlflow.log_metric("Train_Reconstruction_Loss", float(loss['Reconstruction_Loss']))
        mlflow.log_metric("Train_Capacity_Loss", float(loss['Capacity_Loss']))

        if (batch_idx == int(len(self.trainer.datamodule.train_dataset_sub)/self.args.batch_size)-1) and (self.current_epoch % self.args.saveImgAfter == 0):
          save_data(recons, x, odd_recons, batch_idx, self.current_epoch, self.args.test)

        return {'loss': loss['loss'], 'latent_space': latent_space.detach().cpu().numpy(), 'mod_recons': mod_recons}
 
    def training_epoch_end(self, training_step_outputs):
      '''In-built function, mention the processing logic needs to be done after every epoch. Here, original and 
      reconstructed images are getting saved after every saveImgAfter(variable) iteration.'''
      
      path = mlflow.get_artifact_uri()
      print("training_epoch_end mlflow.get_artifact_uri() - ", path)
      path_to_log_artifact = (path.rsplit('/', 1)[0]).split("file://")[1]
      print("path_to_log_artifact - ", path_to_log_artifact)

      #save original latent space
      if not os.path.exists(path_to_log_artifact +'/latent_space/'):
        os.makedirs(path_to_log_artifact +'/latent_space/')

      lt_space = np.zeros((len(training_step_outputs), training_step_outputs[0]['latent_space'].shape[0],  training_step_outputs[0]['latent_space'].shape[1] ))
      for i, ls in enumerate(training_step_outputs):
        lt_space[i] = ls['latent_space']
        
      np.save(os.path.join(path_to_log_artifact +'/latent_space/'+str(self.current_epoch)+'.npy'), lt_space)

      #save odd latent space
      if not os.path.exists(path_to_log_artifact +'/odd_latent_space/'):
        os.makedirs(path_to_log_artifact +'/odd_latent_space/')

      odd_lt_space = np.zeros((len(training_step_outputs), training_step_outputs[0]['mod_recons'].shape[0],  training_step_outputs[0]['mod_recons'].shape[1] ))
      for i, ls in enumerate(training_step_outputs):
        lt_space[i] = ls['mod_recons']

      np.save( os.path.join(path_to_log_artifact +'/odd_latent_space/'+str(self.current_epoch)+'.npy'), lt_space)

      if self.current_epoch % self.args.saveImgAfter == 0:
          #save checkpoint
        if not os.path.exists(path_to_log_artifact +'/checkpoint/'):
          os.makedirs(path_to_log_artifact +'/checkpoint/')
        print(os.path.join(path_to_log_artifact +'/checkpoint/model.pt'))
        torch.save(self.AEModel.state_dict(), os.path.join(path_to_log_artifact +'/checkpoint/model.pt'))


    def test_step(self, batch, batch_idx):
      path = mlflow.get_artifact_uri()
      print("test_step mlflow.get_artifact_uri() - ", path)

      x, y = batch
      out = self(x)
      recons, mu, log_var, latent_space, odd_recons, q, mod_recons = out[0], out[1], out[2], out[3], out[4], out[5], out[6]
      loss = self.AEModel.loss_function(recons, x, mu, log_var, q, batch_idx, self.current_epoch)
      # loss = F.mse_loss(recons, x)   #reduction="sum"
      run_id = mlflow.active_run()
      print("run_id test - ", run_id)
      mlflow.log_metric("Test_Loss", float(loss['loss']))
      mlflow.log_metric("Test_Reconstruction_Loss", float(loss['Reconstruction_Loss']))
      mlflow.log_metric("Test_Capacity_Loss", float(loss['Capacity_Loss']))
      save_data(recons, x, odd_recons, batch_idx, self.current_epoch, self.args.test)
      return {'loss': loss['loss'], 'latent_space': latent_space.detach().cpu().numpy(), 'mod_recons': mod_recons}
 

    def test_epoch_end(self, test_step_outputs):
      path = mlflow.get_artifact_uri()
      print("test_epoch_end mlflow.get_artifact_uri() - ", path)
      path_to_log_artifact = (path.rsplit('/', 1)[0]).split("file://")[1]
      print("path_to_log_artifact - ", path_to_log_artifact)

      #save original latent space
      if not os.path.exists(path_to_log_artifact +'/test_latent_space/'):
        os.makedirs(path_to_log_artifact +'/test_latent_space/')

      print("test_step_outputs shape - ", len(test_step_outputs))
      
      lt_space = np.zeros((len(test_step_outputs), test_step_outputs[0]['latent_space'].shape[0],  test_step_outputs[0]['latent_space'].shape[1] ))
      for i, ls in enumerate(test_step_outputs):
        lt_space[i] = ls['latent_space']
        
      np.save(os.path.join(path_to_log_artifact +'/test_latent_space/'+str(self.current_epoch)+'.npy'), lt_space)

      #save odd latent space
      if not os.path.exists(path_to_log_artifact +'/test_odd_latent_space/'):
        os.makedirs(path_to_log_artifact +'/test_odd_latent_space/')

      odd_lt_space = np.zeros((len(test_step_outputs), test_step_outputs[0]['mod_recons'].shape[0],  test_step_outputs[0]['mod_recons'].shape[1] ))
      for i, ls in enumerate(test_step_outputs):
        lt_space[i] = ls['mod_recons']

      np.save( os.path.join(path_to_log_artifact +'/test_odd_latent_space/'+str(self.current_epoch)+'.npy'), lt_space)

      if self.current_epoch % self.args.saveImgAfter == 0:
          #save checkpoint
        if not os.path.exists(path_to_log_artifact +'/checkpoint/'):
          os.makedirs(path_to_log_artifact +'/checkpoint/')
        print(os.path.join(path_to_log_artifact +'/checkpoint/model.pt'))
          
          
    def configure_optimizers(self):
      '''In-built function, specify the optimiser.'''
      if self.args.optimiser == "AdamW":
          optimiser = torch.optim.AdamW(self.AEModel.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
      elif self.args.optimiser == "SGD":
          optimiser = torch.optim.SGD(self.AEModel.parameters(), self.args.lr)
      else:
          try:
            optimiser = None
            raise ValueError()
          except Exception as ex:
            sys.exit("Possible Optimiser types are - Adam, SGD. Please change the Optimiser input.")
      return optimiser

    # def validation_step(self, batch, batch_idx):
    #   '''In-built function, replaces the validation loop.'''

    #   path = mlflow.get_artifact_uri()
    #   print("validation_step mlflow.get_artifact_uri() - ", path)

    #   x, y = batch
    #   out = self(x)
    #   # recons, inp, mu, log_var = out[0], out[1], out[2], out[3]
    #   # loss_fn()
    #   # loss = F.mse_loss(recons, x)   
    #   # run_id = mlflow.active_run()
    #   # print("run_id train - ", run_id)
    #   # mlflow.log_metric("val_loss", float(loss))

    #   recons, mu, log_var, latent_space, odd_recons, q = out[0], out[1], out[2], out[3], out[4], out[5]
    #   loss = self.AEModel.loss_function(recons, x, mu, log_var, q, batch_idx, self.current_epoch)
    #   # loss = F.mse_loss(recons, x)   #reduction="sum"
    #   run_id = mlflow.active_run()
    #   print("run_id train - ", run_id)
    #   mlflow.log_metric("Val_Loss", float(loss['loss']))
    #   mlflow.log_metric("Val_Reconstruction_Loss", float(loss['Reconstruction_Loss']))
    #   mlflow.log_metric("Val_Capacity_Loss", float(loss['Capacity_Loss']))
    #   # self.log('val_MSEloss_value', loss, prog_bar=True, logger=True) #, on_epoch = True
    #   # self.logger.experiment.log_param(key = "MLFLOW_val_loss", value = float(loss))

    #   #loss = self.ssim_loss(x, recons)
    #   #self.log('val_ssimValue', loss, prog_bar=True, logger=True, on_epoch = True)

    #   # return loss

def save_data(pred, original_img, odd_recons, batchId, current_epoch, test_flag):
  path = mlflow.get_artifact_uri()
  path_to_log_artifact = (path.rsplit('/', 1)[0]).split("file://")[1]
  
  if not os.path.exists(path_to_log_artifact +'/output/'):
    os.makedirs(path_to_log_artifact +'/output/')
  
  ls = [pred[0], original_img[0], odd_recons[0]]
  grid_img = torchvision.utils.make_grid(ls, nrows = 3)

  if test_flag:
    save_image(grid_img, path_to_log_artifact +'/output/'+ str(current_epoch) + '_' + str(batchId)+'_test.png', 'png')
    mlflow.log_image(torch.transpose(torch.transpose(grid_img, 0, 2), 0, 1).detach().cpu().numpy(), str(current_epoch) + '_' + str(batchId)+'_test.png')
  else:
    save_image(grid_img, path_to_log_artifact +'/output/'+ str(current_epoch) + '_' + str(batchId)+'.png', 'png')
    mlflow.log_image(torch.transpose(torch.transpose(grid_img, 0, 2), 0, 1).detach().cpu().numpy(), str(current_epoch) + '_' + str(batchId)+'.png')
  