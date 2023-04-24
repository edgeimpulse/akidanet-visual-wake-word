import sys, os, shutil, signal, random, operator, functools, time, subprocess, math, contextlib, io, skimage, argparse
import logging, threading

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse training scripts')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape',
                    default=os.path.join(dir_path, 'train_input.json'))
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--epochs', type=int, required=False,
                    help='Number of training cycles')
parser.add_argument('--learning-rate', type=float, required=False,
                    help='Learning rate')

args, unknown = parser.parse_known_args()

# Info about the training pipeline (inputs / shapes / modes etc.)
if not os.path.exists(args.info_file):
    print('Info file', args.info_file, 'does not exist')
    exit(1)

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import numpy as np

# Suppress Numpy deprecation warnings
# TODO: Only suppress warnings in production, not during development
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# Filter out this erroneous warning (https://stackoverflow.com/a/70268806 for context)
warnings.filterwarnings('ignore', 'Custom mask layers require a config and must override get_config')

RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

# Since it also includes TensorFlow and numpy, this library should be imported after TensorFlow has been configured
sys.path.append('./resources/libraries')
import ei_tensorflow.training
import ei_tensorflow.conversion
import ei_tensorflow.profiling
import ei_tensorflow.inference
import ei_tensorflow.embeddings
import ei_tensorflow.lr_finder
import ei_tensorflow.brainchip.model
import ei_tensorflow.gpu
from ei_shared.parse_train_input import parse_train_input, parse_input_shape


import json, datetime, time, traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error

input = parse_train_input(args.info_file)

BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'best_model.tf' if input.akidaModel else 'best_model.hdf5')

# Information about the data and input:
# The shape of the model's input (which may be different from the shape of the data)
MODEL_INPUT_SHAPE = parse_input_shape(input.inputShapeString)
# The length of the model's input, used to determine the reshape inside the model
MODEL_INPUT_LENGTH = MODEL_INPUT_SHAPE[0]
MAX_TRAINING_TIME_S = input.maxTrainingTimeSeconds

online_dsp_config = None

if (online_dsp_config != None):
    print('The online DSP experiment is enabled; training will be slower than normal.')

# load imports dependening on import
if (input.mode == 'object-detection' and input.objectDetectionLastLayer == 'mobilenet-ssd'):
    import ei_tensorflow.object_detection

def exit_gracefully(signum, frame):
    print("")
    print("Terminated by user", flush=True)
    time.sleep(0.2)
    sys.exit(1)


def train_model(train_dataset, validation_dataset, input_length, callbacks,
                X_train, X_test, Y_train, Y_test, train_sample_count, classes, classes_values):
    global ei_tensorflow

    override_mode = None
    disable_per_channel_quantization = False
    # We can optionally output a Brainchip Akida pre-trained model
    akida_model = None
    akida_edge_model = None

    if (input.mode == 'object-detection' and input.objectDetectionLastLayer == 'mobilenet-ssd'):
        ei_tensorflow.object_detection.set_limits(max_training_time_s=MAX_TRAINING_TIME_S,
            is_enterprise_project=input.isEnterpriseProject)

    
    import math, random
    import tensorflow as tf
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Reshape
    from tensorflow.keras.optimizers import Adam
    
    from keras import Model
    from keras.layers import Activation, Dropout, Reshape, Flatten
    
    from akida_models.layer_blocks import dense_block
    from akida_models import akidanet_imagenet
    
    from ei_tensorflow import training
    import cnn2snn
    
    BATCH_SIZE = 32
    
    #! Implements the data augmentation policy
    def augmentation_function(input_shape: tuple):
        def augment_image(image, label):
            image = tf.image.random_flip_left_right(image)
    
            #! Increase the image size, then randomly crop it down to
            #! the original dimensions
            resize_factor = random.uniform(1, 1.2)
            new_height = math.floor(resize_factor * input_shape[0])
            new_width = math.floor(resize_factor * input_shape[1])
            image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
            image = tf.image.random_crop(image, size=input_shape)
    
            #! Vary the brightness of the image
            image = tf.image.random_brightness(image, max_delta=0.2)
    
            return image, label
    
        return augment_image
    
    def train(train_dataset: tf.data.Dataset,
              validation_dataset: tf.data.Dataset,
              num_classes: int,
              pretrained_weights: str,
              input_shape: tuple,
              learning_rate: int,
              epochs: int,
              dense_layer_neurons: int,
              dropout: float,
              data_augmentation: bool,
              callbacks,
              alpha: float,
              best_model_path: str,
              quantize_function,
              qat_function,
              edge_learning_function=None,
              additional_classes=None,
              neurons_per_class=None,
              X_train=None):
        #! Create a quantized base model without top layers
        base_model = akidanet_imagenet(input_shape=input_shape,
                                    classes=num_classes,
                                    alpha=alpha,
                                    include_top=False,
                                    input_scaling=None,
                                    pooling='avg')
                                    
         #add gamma constraint
        # from akida_models import add_gamma_constraint
        # base_model = add_gamma_constraint(base_model)
        
        import akida
        
        print(akida.__version__)
        base_model.summary()
    
        base_model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
        
        base_model.summary()
    
        #! Freeze that base model, so it won't be trained
        base_model.trainable = True
    
        output_model = base_model.output
        output_model = Flatten()(output_model)
        if dense_layer_neurons > 0:
            output_model = dense_block(output_model,
                                    units=dense_layer_neurons,
                                    add_batchnorm=False,
                                    add_activation=True)
        if dropout > 0:
            output_model = Dropout(dropout)(output_model)
        output_model = dense_block(output_model,
                                units=num_classes,
                                add_batchnorm=False,
                                add_activation=False)
        output_model = Activation('softmax')(output_model)
        output_model = Reshape((num_classes,))(output_model)
    
        #! Build the model
        model = Model(base_model.input, output_model)
    
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
        if data_augmentation:
            train_dataset = train_dataset.map(augmentation_function(input_shape),
                                              num_parallel_calls=tf.data.AUTOTUNE)
    
        #! This controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
        validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)
    
        #! Train the neural network
        model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
    
        print('')
        print('Initial training done.', flush=True)
        print('')
        
        #! Unfreeze the model before QAT
        model.trainable = False
    
        #! Quantize model to 4/4/8
        akida_model = quantize_function(keras_model=model)
        
        akida_model.summary()
    
        #! Do a quantization-aware training
        akida_model = qat_function(akida_model=akida_model,
                                   train_dataset=train_dataset,
                                   validation_dataset=validation_dataset,
                                   optimizer=Adam(learning_rate=learning_rate),
                                   fine_tune_loss='categorical_crossentropy',
                                   fine_tune_metrics=['accuracy'],
                                   callbacks=callbacks)
        #! Optionally, build the edge learning model
        if edge_learning_function:
            akida_edge_model = edge_learning_function(quantized_model=akida_model,
                                                      X_train=X_train,
                                                      train_dataset=train_dataset,
                                                      validation_dataset=validation_dataset,
                                                      callbacks=callbacks,
                                                      optimizer=opt,
                                                      fine_tune_loss='categorical_crossentropy',
                                                      fine_tune_metrics=['accuracy'],
                                                      additional_classes=additional_classes,
                                                      neurons_per_class=neurons_per_class,
                                                      num_classes=num_classes,
                                                      qat_function=qat_function)
        else:
            akida_edge_model = None
    
    
        return model, akida_model, akida_edge_model
    import tensorflow as tf
    
    
    def akida_quantize_model(
        keras_model,
        weight_quantization: int = 4,
        activ_quantization: int = 4,
        input_weight_quantization: int = 8,
    ):
        import cnn2snn
    
        print("Performing post-training quantization...")
        akida_model = cnn2snn.quantize(
            keras_model,
            weight_quantization=weight_quantization,
            activ_quantization=activ_quantization,
            input_weight_quantization=input_weight_quantization,
        )
        print("Performing post-training quantization OK")
        print("")
    
        return akida_model
    
    
    def akida_perform_qat(
        akida_model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        optimizer: str,
        fine_tune_loss: str,
        fine_tune_metrics: "list[str]",
        callbacks,
        stopping_metric: str = "val_accuracy",
        fit_verbose: int = 2,
        qat_epochs: int = 2#30,
    ):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=stopping_metric,
            mode="max",
            verbose=1,
            min_delta=0,
            patience=10,
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)
    
        print("Running quantization-aware training...")
        akida_model.compile(
            optimizer=optimizer, loss=fine_tune_loss, metrics=fine_tune_metrics
        )
    
        akida_model.fit(
            train_dataset,
            epochs=qat_epochs,
            verbose=fit_verbose,
            validation_data=validation_dataset,
            callbacks=callbacks,
        )
    
        print("Running quantization-aware training OK")
        print("")
    
        return akida_model
    
    
    
    EPOCHS = args.epochs or 20
    LEARNING_RATE = args.learning_rate or 0.0005
    
    # Available pretrained_weights are:
    # akidanet_imagenet_224_alpha_100.h5            - float32 model, 224x224x3, alpha=1.00
    # akidanet_imagenet_224_alpha_50.h5             - float32 model, 224x224x3, alpha=0.50
    # akidanet_imagenet_224_alpha_25.h5             - float32 model, 224x224x3, alpha=0.25
    # akidanet_imagenet_160_alpha_100.h5            - float32 model, 160x160x3, alpha=1.00
    # akidanet_imagenet_160_alpha_50.h5             - float32 model, 160x160x3, alpha=0.50
    # akidanet_imagenet_160_alpha_25.h5             - float32 model, 160x160x3, alpha=0.25
    model, akida_model, akida_edge_model = train(train_dataset=train_dataset,
                                                 validation_dataset=validation_dataset,
                                                 num_classes=classes,
                                                 pretrained_weights='./akidanet_vww.h5',
                                                 input_shape=(96, 96, 3,),
                                                 learning_rate=LEARNING_RATE,
                                                 epochs=EPOCHS,
                                                 dense_layer_neurons=0,
                                                 dropout=0.1,
                                                 data_augmentation=True,
                                                 callbacks=callbacks,
                                                 alpha=0.25,
                                                 best_model_path=BEST_MODEL_PATH,
                                                 quantize_function=akida_quantize_model,
                                                 qat_function=akida_perform_qat,
                                                 edge_learning_function=None,
                                                 additional_classes=None,
                                                 neurons_per_class=None,
                                                 X_train=X_train)
    
    return model, override_mode, disable_per_channel_quantization, akida_model, akida_edge_model

# This callback ensures the frontend doesn't time out by sending a progress update every interval_s seconds.
# This is necessary for long running epochs (in big datasets/complex models)
class BatchLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, train_sample_count, epochs, interval_s = 10):
        # train_sample_count could be smaller than the batch size, so make sure total_batches is atleast
        # 1 to avoid a 'divide by zero' exception in the 'on_train_batch_end' callback.
        self.total_batches = max(1, int(train_sample_count / batch_size))
        self.last_log_time = time.time()
        self.epochs = epochs
        self.interval_s = interval_s

    # Within each epoch, print the time every 10 seconds
    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if self.last_log_time + self.interval_s < current_time:
            print('Epoch {0}% done'.format(int(100 / self.total_batches * batch)), flush=True)
            self.last_log_time = current_time

    # Reset the time the start of every epoch
    def on_epoch_end(self, epoch, logs=None):
        self.last_log_time = time.time()

def main_function():
    """This function is used to avoid contaminating the global scope"""
    classes_values = input.classes
    classes = 1 if input.mode == 'regression' else len(classes_values)

    mode = input.mode
    object_detection_last_layer = input.objectDetectionLastLayer if input.mode == 'object-detection' else None

    train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, has_samples, X_samples, Y_samples = ei_tensorflow.training.get_dataset_from_folder(
        input, args.data_directory, RANDOM_SEED, online_dsp_config, MODEL_INPUT_SHAPE
    )

    callbacks = ei_tensorflow.training.get_callbacks(dir_path, mode, BEST_MODEL_PATH,
        object_detection_last_layer=object_detection_last_layer,
        is_enterprise_project=input.isEnterpriseProject,
        max_training_time_s=MAX_TRAINING_TIME_S,
        enable_tensorboard=input.tensorboardLogging)

    model = None

    print('')
    print('Training model...')
    ei_tensorflow.gpu.print_gpu_info()
    print('Training on {0} inputs, validating on {1} inputs'.format(len(X_train), len(X_test)))
    # USER SPECIFIC STUFF
    model, override_mode, disable_per_channel_quantization, akida_model, akida_edge_model = train_model(train_dataset, validation_dataset,
        MODEL_INPUT_LENGTH, callbacks, X_train, X_test, Y_train, Y_test, len(X_train), classes, classes_values)
    if override_mode is not None:
        mode = override_mode
    # END OF USER SPECIFIC STUFF

    # REST OF THE APP
    print('Finished training', flush=True)
    print('', flush=True)

    # Make sure these variables are here, even when quantization fails
    tflite_quant_model = None

    if mode == 'object-detection':
        tflite_model, tflite_quant_model = ei_tensorflow.object_detection.convert_to_tf_lite(
            args.out_directory,
            saved_model_dir='saved_model',
            validation_dataset=validation_dataset,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite')
    elif mode == 'segmentation':
        from ei_tensorflow.constrained_object_detection.conversion import convert_to_tf_lite
        tflite_model, tflite_quant_model = convert_to_tf_lite(
            args.out_directory, model,
            saved_model_dir='saved_model',
            h5_model_path='model.h5',
            validation_dataset=validation_dataset,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite',
            disable_per_channel=disable_per_channel_quantization)
        if input.akidaModel:
            if not akida_model:
                print('Akida training code must assign a quantized model to a variable named "akida_model"', flush=True)
                exit(1)
            ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_model,
                                                            'akida_model.fbz',
                                                            MODEL_INPUT_SHAPE)
    else:
        model, tflite_model, tflite_quant_model = ei_tensorflow.conversion.convert_to_tf_lite(
            model, BEST_MODEL_PATH, args.out_directory,
            saved_model_dir='saved_model',
            h5_model_path='model.h5',
            validation_dataset=validation_dataset,
            model_input_shape=MODEL_INPUT_SHAPE,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite',
            disable_per_channel=disable_per_channel_quantization,
            syntiant_target=input.syntiantTarget,
            akida_model=input.akidaModel)

        if input.akidaModel:
            if not akida_model:
                print('Akida training code must assign a quantized model to a variable named "akida_model"', flush=True)
                exit(1)

            ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_model,
                                                              'akida_model.fbz',
                                                              MODEL_INPUT_SHAPE)
            if input.akidaEdgeModel:
                ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_edge_model,
                                                                'akida_edge_learning_model.fbz',
                                                                MODEL_INPUT_SHAPE)
            else:
                import os
                model_full_path = os.path.join(args.out_directory, 'akida_edge_learning_model.fbz')
                if os.path.isfile(model_full_path):
                    os.remove(model_full_path)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    main_function()