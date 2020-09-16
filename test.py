# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:14:55 2019

@author: AkhileshAkku
"""

# =============================================================================
import os
import numpy as np
from skimage.io import imread
import pandas as pd
from keras.models import model_from_json
from keras.models import load_model
from glob import glob
#IMG_CHANNELS = 3
def read_and_stack(in_img_list):
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
# 
# =============================================================================
# def convert(val_image_path):
#     final_image_path = os.path.join(val_image_path, val_image_path.split('\\')[-2]+'.png')
#     return final_image_path
# 
# =============================================================================
   
# val_image_path = input('input the name of the image file without (.png) extension) : ') #make sure the image(jpg or png file) is also included in the path.
# x = str(val_image_path)+'.png'

# =============================================================================
# final_image_path = convert(val_image_path)
# print(final_image_path)

val_image_path = 'val_image'
x = val_image_path + '.png'
y = x.split()


out_df = pd.DataFrame({'ImageName': [y], 'ImageId': [val_image_path]})
IMG_CHANNELS = 3
out_df['ImageName'] = out_df['ImageName'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS]) 

import matplotlib.pyplot as plt

json_file = open('nucleus_model.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("nuclei_model_weights.h5")
print("Loaded model from disk")

out_df['mask'] = out_df['ImageName'].map(lambda x: loaded_model.predict(np.expand_dims(x, 0))[0, :, :, 0])
print(out_df['mask'])
fig, m_axs = plt.subplots(1, 1, figsize = (12, 6))

m_axs.imshow(out_df.sample(1)['mask'][0])
m_axs.axis('off')
m_axs.set_title('Prediction')

# for (_, d_row), (c_im) in zip(out_df.sample(1).iterrows(), 
#                                      m_axs):
#     c_im.imshow(d_row['mask'])
#     c_im.axis('off')
#     c_im.set_title('Prediction')
plt.show()
