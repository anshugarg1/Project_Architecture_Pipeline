from torchvision import transforms as tf

def transformImg(img_h, img_w):
  trfm = tf.Compose([tf.Resize((img_h,img_w)), tf.ToTensor(), tf.Normalize([0.3315, 0.3527, 0.4412], [0.3075, 0.3239, 0.3370])]) 
  return trfm

