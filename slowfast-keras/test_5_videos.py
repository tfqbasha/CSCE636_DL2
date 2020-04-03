"""**TEST_CELL**"""

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
from dataset.dataset import DataGenerator
from sklearn.metrics import accuracy_score, f1_score
class Args:
  root_path = '../../CSCE636_DL2/slowfast-keras'
  video_path = '../../Data/data_clips_100_2class_jpg'
  name_path = '../../Data/data_clips_100_2class_jpg/classInd.txt'
  #name_path = 'data_clips_100_2class_jpg2/classInd.txt'
  train_list = '../../Data/data_clips_100_2class_jpg/train_new.txt'
  val_list = '../../Data/data_clips_100_2class_jpg/test_new.txt'
  result_path = 'results_0104_2'
  data_name = 'ntu'
  gpus = [1]
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
  batch_size = 4
  epochs = 30
  workers = 4
  network = 'resnet50'
  pretrained_weights = None
  test_videos_path = '../../Data/data_clips_100_ref_test_jpg'
  test_list_path = '../../Data/data_clips_100_ref_test_jpg/test_final.txt'
  #test_list_path = '../../Data/data_clips_100_2class_jpg/train_new.txt'
  #test_videos_path = '../../Data/data_clips_100_2class_jpg'

  #test_list_path = 'data_clips_100_2class_jpg2/train_new.txt'
  #test_videos_path = 'data_clips_100_2class_jpg2'

opt = Args()
# load model
model = tf.keras.models.load_model('SlowFast_refrigerator_0104_2.h5')
# summarize model.
#model.summary()
predict_data_generator = DataGenerator(opt.data_name, opt.test_videos_path, opt.test_list_path, opt.name_path, 
                                        'val', 1, opt.num_classes, False, opt.short_side, 
                                        opt.crop_size, opt.clip_len, opt.n_samples_for_each_video, to_fit=False)  
model_predicted = model.predict(predict_data_generator)
print(type(model_predicted))
import numpy as np
#np.savetxt('test_0104_2.txt', model_predicted)
print(model_predicted)
k1 = np.argmax(model_predicted, axis=1)
print('k1 shape:', type(k1), k1.shape)
k2 = np.array([0, 0, 0, 0, 0, 1, 1, 1 ])
print('k2 shape:', type(k2), k2.shape)
print('predicted labels:', k1)
print('true labels:', k2)
print('label info: opening_refrigerator:0 and other_classes:1')

#test_pred = model.predict_classes(X_b_shaped)
#k1 = np.array(test_pred[:,0])
#k2 = np.array(y_b)
#for i in range(len(k2)):
  #    if k1[i] == k2[i]:
   #           count = count+1
    #          accuracy = (count*100)/len(y_b)
acc = accuracy_score(k2, k1)
f1 = f1_score(k2, k1)
print('The accuracy of this model is ' + str(acc))
print('The f1 score of this model is ' + str(f1))

