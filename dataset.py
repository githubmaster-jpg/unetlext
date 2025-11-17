from torch.utils.data import Dataset
from torch.utils.tensorboard.summary import image
import numpy as np
import cv2

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #  H.W
    xray = xray.astype(np.float32) / 255.0
    # Note !
    x, y = xray.shape
    xray = xray.reshape((1, x, y)) # 1.H.W
    return xray

def read_mask(path):
    try:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #  H.W
    except:
        return None
    # Note!
    # Note!
    mask = (mask > 0).astype(np.float32)
    x, y = mask.shape
    mask = mask.reshape((1, x, y)) # 1.H.W
    return mask

class Knee_dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        xray = read_xray(self.df['xrays'].iloc[index]) #self.df.xrays.iloc[index]
        mask = read_mask(self.df['masks'].iloc[index])
        if mask is None:
            results = {
                'xray': xray
            }
        else:
            results = {
                'xray': xray,
                'mask': mask
            }   
        return results