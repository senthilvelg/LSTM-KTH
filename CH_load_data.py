import cv2
import io
import os
import subprocess
import glob
import numpy as np
#base_dir = "/Users/madhuhegde/Downloads/cholec80/"
#image_dir = base_dir+"images/"
#label_dir = base_dir+"labels/"

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}


def load_cholec_data(image_dir, label_dir):

  image_files = glob.glob(image_dir+"*.jpg")
  label_files = glob.glob(label_dir+"*.txt")

  classes_array = list()
  data_array = list()


  for label_file in label_files:
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      for line in handle:
        class_label = class_labels[line.split('\t')[1].strip()]
        classes_array.append(class_label)
        
        
#print(classes_array)   
#print(len(image_files), len(classes_array))
  #it was found that len(image_file) is one more than number of labels
  array_len = min(len(image_files), len(classes_array))
  
  image_files = image_files[0:array_len]
  classes_array = classes_array[0:array_len]
  #print(len(image_files), len(classes_array))

  classes_one_hot = np.zeros((len(classes_array), len(class_labels)))
  classes_one_hot[np.arange(len(classes_array)), classes_array] = 1  

  for image_file in image_files:
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = (image-128.0)/128.0;
    data_array.append(image) 
  
  data_array = np.array(data_array)
  #print(np.max(data_array[:,:,:,1:100]), np.min(data_array[:,:,:,1:100]))
  print(data_array.shape, classes_one_hot.shape)  
  return(data_array, classes_one_hot)


