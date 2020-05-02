#!pip install tensorflow==1.15.0
#!python3 -c 'import tensorflow as tf; print(tf.__version__)'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model
from model import nets
from model import slowfast
from opts import parse_opts
from utils import get_optimizer, SGDRScheduler_with_WarmUp, TrainPrint, PrintLearningRate, ParallelModelCheckpoint
from dataset.dataset_new2 import DataGenerator
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np
import os
from dataset.utils import load_value_file
import math
import json
import matplotlib.pyplot as plt

#from tensorflow.python.keras.layers import Layer, InputSpec
dir=$(pwd)
class Args:
  root_path = '$dir/slowfast-keras'
  #video_path = '/content/drive/My Drive/Colab Notebooks/CSCE636/Data/data_clips_100_ref_jpg'
  name_path = '$dir/test_jpg/classInd.txt'
  #train_list = '/content/drive/My Drive/Colab Notebooks/CSCE636/Data/data_clips_100_ref/train.txt'
  #val_list = '/content/drive/My Drive/Colab Notebooks/CSCE636/Data/data_clips_100_ref/test.txt'
  result_path = 'results'
  data_name = 'ntu'
  gpus = [0]
  log_dir = 'log'
  num_classes = 2
  crop_size = 224
  clip_len = 64
  short_side = [256, 320]
  n_samples_for_each_video = 1
  lr = 0.00001
  momentum = 0.9
  weight_decay = 1e-4
  lr_decay = 0.8
  cycle_length = 10
  multi_factor = 1.5
  warm_up_epoch = 5
  optimizer = 'SGD'
  batch_size = 1
  epochs = 30
  workers = 5
  network = 'resnet50'
  pretrained_weights = None
  test_list_path = '$dir/test_jpg/test.txt'
  test_videos_path = '$dir/test_jpg'
  split_frames_for_test = 6

opt = Args()

for path_label in open(opt.test_list_path, 'r'):
    path, _ = path_label.split()
    path, _ = os.path.splitext(path)
    full_video_path = os.path.join(opt.test_videos_path , path)
    print('full_path', full_video_path)
    n_frame_path = os.path.join(full_video_path, 'n_frames')
    n_frames = int(load_value_file(n_frame_path))
    print('n_frames:', n_frames)

predict_data_generator = DataGenerator(opt.data_name, opt.test_videos_path, opt.test_list_path, opt.name_path, 
                                        'val', 1, opt.num_classes, False, opt.short_side, 
                                        opt.crop_size, opt.clip_len, opt.n_samples_for_each_video, False, opt.split_frames_for_test) 

# load model
model =  tf.keras.models.load_model('SlowFast_refrigerator_2904.h5')
model_predicted = model.predict(predict_data_generator)
#model_evaluated = model.evaluate_generator(predict_data_generator)
#print("Evaluate 2604: Loss and Accuracy", model_evaluated)
print("predicted 2704:", model_predicted, type(model_predicted))
model_predicted_argmax = np.argmax(model_predicted, axis=1)
print("predicted 2704:", model_predicted_argmax, type(model_predicted_argmax))
print("predicted_ravel", model_predicted[4][0], type(np.ravel(model_predicted_argmax)) )

frame_step_size = math.ceil(n_frames/opt.split_frames_for_test)
fileName = 'Open or Close Refrigerator'
i = 0
fileJson = {}
fileJson[fileName] = []
x_for_plot = []
y_for_plot = []
video_length = 0.39
time_per_frame = 0.39/n_frames

while(i<opt.split_frames_for_test):
  time_step = round(frame_step_size*time_per_frame,3)
  start_frame = i*time_step
  end_frame   = (i+1)*time_step
  x_for_plot.append(end_frame)
  y_for_plot.append(model_predicted[i][0])
  fileJson[fileName].append([str(start_frame), str(end_frame ), str(model_predicted_argmax[i]) ] )
  i += 1

with open("529005214.json", "w") as outfile: 
    json.dump(fileJson, outfile) 

#y_for_plot = model_predicted_argmax

fig = plt.step(x_for_plot, y_for_plot)
plt.xlabel('Video length (in terms of frame number)')
plt.ylabel('Prediction')
plt.xticks(np.arange(0, video_length,  round(frame_step_size*time_per_frame, 2) ))
plt.savefig('529005214.png')

