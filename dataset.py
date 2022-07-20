from torchvision import datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tf
import torch
import os
import glob
from PIL import Image
import cv2

from definitions import mode_folder_mapping, night_folder_list, alternate_route_folder_list, snow_folder_list, rain_folder_list, sun_folder_list
 
class OODCustomDataset(Dataset):
  def __init__(self, data_mode, rootPath:str = "/storage/Oxford_Robotcar/", subPath:str = "stereo/centre_colored/", 
  test_flag:bool = False, transform = None):

    self.rootPath = rootPath   
    self.transform = transform
    self.subPath = subPath
    self.imageList = []
    self.count = 0
    self.imageFolderList = []
    self.imageCountPerFolder = []
    self.imgNameList = []
    self.imgExtension = ".png"
    self.test_flag = test_flag
    self.data_mode = data_mode
    
    all_folders = mode_folder_mapping[data_mode]

    for folder in all_folders: 
      print("inside folder - ", folder)
      fpath = os.path.join(rootPath, folder)
      if os.path.isdir(fpath):
        fullPath = os.path.join(fpath, subPath)
        
        self.imageCountPerFolder.append(len(os.listdir(fullPath))) #store count of images in each folder.
        self.imageFolderList.append(fullPath) #stores the folder names inside /storage/Oxford_Robotcar/ for example, 2014-05-06-12-54-54, 2014-07-14-15-16-36 etc
        self.imgNameList.append(os.listdir(fullPath)) #stores image name present in the each folder
    
  def __len__(self):
    return sum(imgCount for imgCount in self.imageCountPerFolder)
  
  def __getitem__(self, idx):
    '''In-built function, Return the image to Dataloader as per the index.'''

    imgCount = 0
    for i, count in enumerate(self.imageCountPerFolder):  
      imgCount += count 
      if imgCount > idx:
        imgCount = imgCount-count
        break
    imgIndex = idx - imgCount  # this will provide the image name 
    folderIndex = i
    sample = self.transform(Image.open(os.path.join(self.imageFolderList[i], self.imgNameList[folderIndex][imgIndex])))
    return sample, 0
    

#329744