import os
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model
from model import nets
from opts import parse_opts
from utils import get_optimizer, SGDRScheduler_with_WarmUp, TrainPrint, PrintLearningRate, ParallelModelCheckpoint
from dataset.dataset import DataGenerator


def create_callbacks(opt, steps_per_epoch, model=None):
    log_dir = os.path.join(opt.root_path, opt.log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True)

    result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if model is not None:
        print("mbashat: entered create_callbacks if model")
        checkpoint = ParallelModelCheckpoint(model, os.path.join(result_path, '{epoch:03d}.h5'),
                                    monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)
    else:
        print("mbashat: entered create_callbacks if not model")
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
    predict_data_generator = DataGenerator(opt.data_name, opt.test_videos_path, opt.test_list_path, opt.name_path, 
                                        'val', 1, opt.num_classes, False, opt.short_side, 
                                        opt.crop_size, opt.clip_len, opt.n_samples_for_each_video, to_fit=False)  
    
    
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
        #history = model.fit_generator(train_data_generator, steps_per_epoch=max(1, train_data_generator.__len__()),
                            #epochs=opt.epochs, validation_data=val_data_generator, validation_steps=max(1, val_data_generator.__len__()),
                            #workers=opt.workers, callbacks=callbacks)
        history = model.fit_generator(train_data_generator, steps_per_epoch=max(1, train_data_generator.__len__()), epochs=opt.epochs, validation_data=val_data_generator, validation_steps=max(1, val_data_generator.__len__()),
                        workers=opt.workers)
    print("model fit done")
    print("mbashat: saving model")
    model.save('SlowFast_refrigerator.h5')
    print("mbashat: saving weights at :", os.path.join(os.path.join(opt.root_path, opt.result_path) ))
    model.save_weights(os.path.join(os.path.join(opt.root_path, opt.result_path), 'trained_weights_final.h5'))
    print("mbashat: predict on val data")
    model_predicted = model.predict(predict_data_generator)
    import numpy as np
    print(np.argmax(model_predicted, axis=1))
    print("mbashat: saving weights to variable")
    #for layer in model.layers:
      #weights = layer.get_weights()
      #print(weights)
    #for video_batch, labels_batch in train_data_generator:
      #print('video batch shape, label shape:', video_batch.shape, labels_batch)
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    print("end of code")


if __name__=="__main__":
    opt = parse_opts()
    print(opt)
    if len(opt.gpus) > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, opt.gpus))
    train(opt)
