import os
import math
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
import csv
import matplotlib.pyplot as plt
import numpy as np

class Args:
  root_path = '../../CSCE636_DL2/slowfast-keras'
  video_path = 'data_clips_100_2class_jpg2'
  name_path = 'data_clips_100_2class_jpg2/classInd.txt'
  train_list = 'data_clips_100_2class_jpg2/train_new.txt'
  val_list = 'data_clips_100_2class_jpg2/val_new.txt'
  result_path = 'results_0204_3'
  data_name = 'ntu'
  gpus = [1]
  log_dir = 'log_0204_3'
  num_classes = 2
  crop_size = 224
  clip_len = 64
  short_side = [256, 320]
  n_samples_for_each_video = 1
  lr = 0.000001
  momentum = 0.9
  weight_decay = 1e-4
  lr_decay = 0.8
  cycle_length = 10
  multi_factor = 1.5
  warm_up_epoch = 5
  optimizer = 'SGD'
  batch_size = 8
  epochs = 30
  workers = 4
  network = 'resnet50'
  pretrained_weights = None
  #test_videos_path = '/content/drive/My Drive/Colab Notebooks/CSCE636/Data/data_clips_100_ref_test_jpg'
  #test_list_path = '/content/drive/My Drive/Colab Notebooks/CSCE636/Data/data_clips_100_ref/test_final.txt'


def create_callbacks(opt, steps_per_epoch, model=None):
    log_dir = os.path.join(opt.root_path, opt.log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True)

    result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if model is not None:
        print("mbashat: using ParallelModel")
        checkpoint = ParallelModelCheckpoint(model, os.path.join(result_path, '{epoch:03d}.h5'),
                                    monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)
    else:
        print("mbashat: using ModelCheckpoint")
        checkpoint = ModelCheckpoint(os.path.join(result_path, '{epoch:03d}.h5'),
                                    monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10)
    learning_rate_scheduler = SGDRScheduler_with_WarmUp(0, opt.lr, steps_per_epoch, lr_decay=opt.lr_decay, 
                                                        cycle_length=opt.cycle_length, multi_factor=opt.multi_factor,
                                                        warm_up_epoch=opt.warm_up_epoch)
    #training_print = TrainPrint(steps_per_epoch, opt.epochs)
    print_lr = PrintLearningRate()

    return [tensorboard, learning_rate_scheduler, print_lr, checkpoint, early_stopping]
    


def train(opt):
    K.clear_session()
    video_input = Input(shape=(None, None, None, 3))
    model = nets.network[opt.network](video_input, num_classes=opt.num_classes)
    print("Create {} model with {} classes".format(opt.network, opt.num_classes))

    if opt.pretrained_weights is not None:
        model.load_weights(opt.pretrained_weights)
        print("Loading weights from {}".format(opt.pretrained_weights))

    optimizer = get_optimizer(opt)

    train_data_generator = DataGenerator(opt.data_name, opt.video_path, opt.train_list, opt.name_path, 
                                        'train', opt.batch_size, opt.num_classes, True, opt.short_side, 
                                        opt.crop_size, opt.clip_len, opt.n_samples_for_each_video)                                     
    val_data_generator = DataGenerator(opt.data_name, opt.video_path, opt.val_list, opt.name_path, 'val', 
                                        opt.batch_size, opt.num_classes, False, opt.short_side, 
                                        opt.crop_size, opt.clip_len, opt.n_samples_for_each_video)
    #predict_data_generator = DataGenerator(opt.data_name, opt.test_videos_path, opt.test_list_path, opt.name_path, 
     #                                   'val', 1, opt.num_classes, False, opt.short_side, 
      #                                  opt.crop_size, opt.clip_len, opt.n_samples_for_each_video, to_fit=False)  
    
    
    callbacks = create_callbacks(opt, max(1, train_data_generator.__len__()), model)

    if len(opt.gpus) > 1:
        print('Using multi gpus')
        parallel_model = multi_gpu_model(model, gpus=len(opt.gpus))
        parallel_model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
        parallel_model.fit_generator(train_data_generator, steps_per_epoch=max(1, train_data_generator.__len__()),
                            epochs=opt.epochs, validation_data=val_data_generator, validation_steps=max(1, val_data_generator.__len__()),
                            workers=opt.workers, callbacks=callbacks)
    else:
        print("GPU <1 run")
        model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
        print("compile done")
        print("mbashat : Val_steps, steps_per_epoch, Val_dat", max(1, val_data_generator.__len__()), max(1, train_data_generator.__len__()), val_data_generator)
        global history_mbt
        #history_mbt = model.fit_generator(train_data_generator, steps_per_epoch=max(1, train_data_generator.__len__()),
                            #epochs=opt.epochs, validation_data=val_data_generator, validation_steps=max(1, val_data_generator.__len__()),
                            #workers=opt.workers, callbacks=callbacks)
        history_mbt = model.fit_generator(train_data_generator, steps_per_epoch=max(1, train_data_generator.__len__()), epochs=opt.epochs, validation_data=val_data_generator, validation_steps=max(1, val_data_generator.__len__()),
                        workers=opt.workers)
        
        #model.summary()
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)
    print("model fit done")
    print("mbashat: saving model")
    model.save('SlowFast_refrigerator_0204_3.h5')
    #print("mbashat: saving weights at :", os.path.join(os.path.join(opt.root_path, opt.result_path) ))
    #model.save_weights(os.path.join(os.path.join(opt.root_path, opt.result_path), 'trained_weights_final_200.h5'))
    #print("mbashat: predict on val data")
    #model_predicted = model.predict(predict_data_generator)
    #import numpy as np
    #print(np.argmax(model_predicted, axis=1))
    #print("mbashat: saving weights to variable")
    #for layer in model.layers:
      #weights = layer.get_weights()
      #print(weights)
    #for video_batch, labels_batch in train_data_generator:
      #print('video batch shape, label shape:', video_batch.shape, labels_batch)
    print('printking Keys in history')
    for key in history_mbt.history:
      print(key)
    #%matplotlib inline
    acc = history_mbt.history['acc']
    val_acc = history_mbt.history['val_acc']
    loss = history_mbt.history['loss']
    val_loss = history_mbt.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    #plt.figure()
    plt.show()
    plt.savefig('accuracy_0204_3.png')
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig('loss_0204_3.png')
    np.savetxt('loss_acc_0204_3.txt', history_mbt)
    with open('loss_acc_0204_3.csv', 'w') as f:
            for key in history_mbt.history():
                        f.write("%s,%s\n"%(key,history_mbt[key]))
    f.close()
    print("end of code")


if __name__=="__main__":
    opt = Args()
    print(opt)
    if len(opt.gpus) > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, opt.gpus))
    train(opt)

