import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import tqdm

def print_auto_logged_info(r):
  '''Function to print all the parameters information stored by MLFlow.'''
  tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
  artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
  print("run_id: {}".format(r.info.run_id))
  print("artifacts: {}".format(artifacts))
  print("params: {}".format(r.data.params))
  print("metrics: {}".format(r.data.metrics))
  print("tags: {}".format(tags))
  
    
def mean_std_cal(loader, datasetLength):
  '''Function to calculate normalisation parameters (mean, standard deviation).'''
  img = next(iter(loader))
  print("Img shape: ", img[0].shape)
  print("Dataset Length:", datasetLength)
  
  psum = torch.tensor([0.0,0.0,0.0])
  psum_sq = torch.tensor([0.0,0.0,0.0])
  
  for inputs in iter(loader):
    # print(type(inputs))
    # print(inputs[0].shape)
    # psum += torch.sum(inputs[0], dim = (0,2,3))  # psum will have sum seperately for all channels i.e. dim = 1 
    # psum_sq += torch.sum(inputs[0]** 2, dim = (0,2,3))

    psum += torch.sum(inputs[0], dim = (1,2))  # psum will have sum seperately for all channels i.e. dim = 1 
    psum_sq += torch.sum(inputs[0]** 2, dim = (1,2))
  
  # total = datasetLength * img[0].shape[2] * img[0].shape[3]
  total = datasetLength * img[0].shape[1] * img[0].shape[2]

  total_mean = psum/total
  total_var = (psum_sq/total) - (total_mean ** 2)
  total_std = torch.sqrt(total_var)
  print("total_mean - ", total_mean)
  print("total_std - ", total_std)