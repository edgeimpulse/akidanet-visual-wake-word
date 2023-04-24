###
###  Note: This is a duplicate of the logic used in Studio. It is used by the Tuner. They will be merged eventually.
###
from __future__ import print_function
import edge_impulse_sdk
import asyncio
import json
import re
import socketio
import time
import traceback
import os
from edge_impulse_sdk.rest import ApiException
from pprint import pprint
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse
from kerastuner.protos import service_pb2
from kerastuner.protos import kerastuner_pb2
from kerastuner.engine import trial as trial_module
import uuid
import concurrent.futures
import threading
# from train import get_model_metadata
import numpy as np
import tensorflow as tf
import json, datetime, time, traceback
import sys, os, shutil, signal, random, operator, functools, time, subprocess, math, contextlib, io
import logging, threading
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss, mean_squared_error
import ei_tensorflow.inference

def tflite_predict(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.as_numpy_iterator():
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.expand_dims(item_as_tensor, 0)
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        scores = ei_tensorflow.inference.process_output(output_details, output)
        pred_y.append(scores)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def get_tensor_details(tensor):
    """Obtains the quantization parameters for a given tensor"""
    details = {
        'dataType': None,
        'name': tensor['name'],
        'shape': tensor['shape'].tolist(),
        'quantizationScale': None,
        'quantizationZeroPoint': None
    }
    if tensor['dtype'] is np.int8:
        details['dataType'] = 'int8'
        details['quantizationScale'] = tensor['quantization'][0]
        details['quantizationZeroPoint'] = tensor['quantization'][1]
    elif tensor['dtype'] is np.float32:
        details['dataType'] = 'float32'
    else:
        raise Exception('Model tensor has an unknown datatype, ', tensor['dtype'])

    return details


def get_io_details(model, model_type):
    """Gets the input and output datatype and quantization details for a model"""
    interpreter = tf.lite.Interpreter(model_content=model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = list(map(get_tensor_details, input_details))
    outputs = list(map(get_tensor_details, output_details))

    return {
        'modelType': model_type,
        'inputs': inputs,
        'outputs': outputs
    }

def profile_tflite_model(model_type, model, model_file, validation_dataset, Y_test, test_dataset, Y_real_test, train_dataset, Y_train, memory, mode, prepare_model_tf_lite=None, prepare_model_tf_lite_eon=None):
    """Calculates performance statistics for a TensorFlow Lite model"""

    prediction = tflite_predict(model, validation_dataset, len(Y_test))
    if test_dataset != None:
        prediction_test = tflite_predict(model, test_dataset, len(Y_real_test))
    prediction_train = tflite_predict(model, train_dataset, len(Y_train))

    if mode == 'classification':
        matrix = confusion_matrix(Y_test.argmax(axis=1), prediction.argmax(axis=1))
        report = classification_report(Y_test.argmax(axis=1), prediction.argmax(axis=1), output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        loss = log_loss(Y_test, prediction)

        if test_dataset != None:
            matrix_test = confusion_matrix(Y_real_test.argmax(axis=1), prediction_test.argmax(axis=1))
            report_test = classification_report(Y_real_test.argmax(axis=1), prediction_test.argmax(axis=1), output_dict=True, zero_division=0)
            accuracy_test = report_test['accuracy']
            loss_test = log_loss(Y_real_test, prediction_test)

        matrix_train = confusion_matrix(Y_train.argmax(axis=1), prediction_train.argmax(axis=1))
        report_train = classification_report(Y_train.argmax(axis=1), prediction_train.argmax(axis=1), output_dict=True, zero_division=0)
        accuracy_train = report_train['accuracy']
        loss_train = log_loss(Y_train, prediction_train)

    elif mode == 'regression':
        matrix = np.array([])
        report = {}
        accuracy = 0
        loss = mean_squared_error(Y_test, prediction[:,0])

    find_arena_size_output = ''

    if not memory and (mode == 'object-detection' or mode == 'yolov5' or mode == 'yolov5v5-drpai' or mode == 'yolox' or mode == 'yolov7'):
        memory = {}
        memory['tflite'] = {
            'ram': 0,
            'rom': os.stat(model_file).st_size,
            'arenaSize': 0,
            'modelSize': os.stat(model_file).st_size
        }
        memory['eon'] = {
            'ram': 0,
            'rom': os.stat(model_file).st_size,
            'arenaSize': 0,
            'modelSize': os.stat(model_file).st_size
        }
    elif not memory:
        memory = {}

        try:
            with open('prepare_tflite.sh', 'w') as f:
                f.write(prepare_model_tf_lite.replace('placeholder', model_file))
            with open('prepare_eon.sh', 'w') as f:
                f.write(prepare_model_tf_lite_eon.replace('placeholder', model_file))

            print('Profiling ' + model_type + ' model (tflite)...')

            if os.path.exists('/app/benchmark/tflite-model'):
                shutil.rmtree('/app/benchmark/tflite-model')
            subprocess.check_output(['sh', 'prepare_tflite.sh']).decode("utf-8")
            tflite_output = json.loads(subprocess.check_output(['/app/benchmark/benchmark.sh', '--tflite-type',
                'float32',
                '--tflite-file', model_file
                ]).decode("utf-8"))
            if os.getenv('K8S_ENVIRONMENT') == 'staging' or os.getenv('K8S_ENVIRONMENT') == 'test':
                print(tflite_output['logLines'])

            # Add fudge factor since the target architecture is different (only on TFLite)
            old_arena_size = tflite_output['arenaSize']

            extra_arena_size = int(math.floor((math.ceil(old_arena_size) * 0.2) + 1024))
            tflite_output['ram'] = tflite_output['ram'] + extra_arena_size
            tflite_output['arenaSize'] = tflite_output['arenaSize'] + extra_arena_size
            memory['tflite'] = tflite_output

            print('Profiling ' + model_type + ' model (EON)...')

            if os.path.exists('/app/benchmark/tflite-model'):
                shutil.rmtree('/app/benchmark/tflite-model')
            subprocess.check_output(['sh', 'prepare_eon.sh']).decode("utf-8")
            eon_output = json.loads(subprocess.check_output(['/app/benchmark/benchmark.sh', '--tflite-type',
                'float32',
                '--tflite-file', model_file,
                '--eon'
                ]).decode("utf-8"))
            if os.getenv('K8S_ENVIRONMENT') == 'staging' or os.getenv('K8S_ENVIRONMENT') == 'test':
                print(eon_output['logLines'])

            memory['eon'] = eon_output
        except Exception as err:
            print('Error while finding memory:', flush=True)
            print(err)
            traceback.print_exc()
            memory = None

    res = {
        'type': model_type,
        'loss': loss,
        'accuracy': accuracy,
        'confusionMatrix': matrix.tolist(),
        'report': report,
        'lossTrain': loss_train,
        'accuracyTrain': accuracy_train,
        'confusionMatrixTrain': matrix_train.tolist(),
        'reportTrain': report_train,
        'size': len(model),
        'estimatedMACCs': None,
        'memory': memory
    }

    if test_dataset != None:
        res['lossTest'] = loss_test
        res['accuracyTest'] = accuracy_test
        res['confusionMatrixTest'] = matrix_test.tolist()
        res['reportTest'] = report_test

    return res

# Useful reference: https://machinethink.net/blog/how-fast-is-my-model/
def estimate_maccs_for_layer(layer):
    """Estimate the number of multiply-accumulates in a given Keras layer."""
    """Better than flops because there's hardware support for maccs."""
    if isinstance(layer, tf.keras.layers.Dense):
        # Ignore the batch dimension
        input_count = functools.reduce(operator.mul, layer.input.shape[1:], 1)
        return input_count * layer.units

    if (isinstance(layer, tf.keras.layers.Conv1D)
        or isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.Conv3D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        # The channel is either at the start or the end of the shape (ignoring)
        # the batch dimension
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
        else:
            input_channels = layer.input.shape[-1]
        # Ignore the batch dimension but include the channels
        output_size = functools.reduce(operator.mul, layer.output.shape[1:])
        return kernel_size * input_channels * output_size

    if (isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.DepthwiseConv2D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
            output_channels = layer.output.shape[1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[2:])
        else:
            input_channels = layer.input.shape[-1]
            output_channels = layer.output.shape[-1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[1:-1])
        # Calculate the MACCs for depthwise and pointwise steps
        depthwise_count = kernel_size * input_channels * output_size
        # If this is just a depthwise conv, we can return early
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return depthwise_count
        # Otherwise, calculate MACCs for the pointwise step and add them
        pointwise_count = input_channels * output_size * output_channels
        return depthwise_count + pointwise_count

    if isinstance(layer, tf.keras.Model):
        return estimate_maccs_for_model(layer)

    # For other layers just return 0. These are mostly stuff that doesn't involve MACCs
    # or stuff that isn't supported by TF Lite for Microcontrollers yet.
    return 0

def estimate_maccs_for_model(keras_model):
    maccs = 0
    for layer in keras_model.layers:
        try:
            layer_maccs = estimate_maccs_for_layer(layer)
            maccs += layer_maccs
        except Exception as err:
            print('Error while estimating maccs for layer')
            print(err)
    return maccs

def describe_layers(keras_model):
    layers = []
    for l in range(len(keras_model.layers)):
        layer = keras_model.layers[l]
        input = layer.input
        if isinstance(input, list):
            input = input[0]
        layers.append({
            'input': {
                'shape': input.shape[1],
                'name': input.name,
                'type': str(input.dtype)
            },
            'output': {
                'shape': layer.output.shape[1],
                'name': layer.output.name,
                'type': str(layer.output.dtype)
            }
        })

    return layers

def get_recommended_model_type(float32_perf, int8_perf):
    # For now, always recommend int8 if available
    if int8_perf:
        return 'int8'
    else:
        return 'float32'

def get_model_metadata(keras_model, validation_dataset, Y_test, test_dataset, Y_real_test, train_dataset, Y_train, class_names, curr_metadata, model_float32=None, model_int8=None, file_float32=None, file_int8=None, mode=None, prepare_model_tf_lite=None, prepare_model_tf_lite_eon=None):
    metadata = {
        'metadataVersion': 4,
        'created': datetime.datetime.now().isoformat(),
        'layers': describe_layers(keras_model),
        'classNames': class_names,
        'availableModelTypes': [],
        'recommendedModelType': '',
        'modelValidationMetrics': [],
        'modelIODetails': [],
        'mode': mode
    }

    # For now we'll calculate this based on the Keras model; in the future we may want to
    # do it for each individual generated model type.
    estimated_maccs = estimate_maccs_for_model(keras_model)

    has_no_changed_layers = (curr_metadata and
        'layers' in curr_metadata and
        'metadataVersion' in curr_metadata and
        curr_metadata['metadataVersion'] == metadata['metadataVersion'] and
        json.dumps(metadata['layers']) == json.dumps(curr_metadata['layers']))

    float32_perf = None
    int8_perf = None

    if model_float32:
        try:
            model_type = 'float32'

            # don't reprofile RAM/ROM if no changed layers
            memory = None
            # if (has_no_changed_layers):
            #     curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
            #     if (len(curr_metrics) > 0):
            #         memory = curr_metrics[0]['memory']

            float32_perf = profile_tflite_model(model_type, model_float32, file_float32, validation_dataset, Y_test, test_dataset, Y_real_test, train_dataset, Y_train, memory, mode, prepare_model_tf_lite, prepare_model_tf_lite_eon)
            float32_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(float32_perf)
            metadata['modelIODetails'].append(get_io_details(model_float32, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite float32 model:')
            print(err)
            traceback.print_exc()

    if model_int8:
        try:
            model_type = 'int8'

            # don't reprofile RAM/ROM if no changed layers
            memory = None
            # if (has_no_changed_layers):
            #     curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
            #     if (len(curr_metrics) > 0):
            #         memory = curr_metrics[0]['memory']

            int8_perf = profile_tflite_model(model_type, model_int8, file_int8, validation_dataset, Y_test, test_dataset, Y_real_test, train_dataset, Y_train, memory, mode, prepare_model_tf_lite, prepare_model_tf_lite_eon)
            int8_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(int8_perf)
            metadata['modelIODetails'].append(get_io_details(model_int8, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite int8 model:')
            print(err)
            traceback.print_exc()

    # Decide which model to recommend
    recommended_model_type = get_recommended_model_type(float32_perf, int8_perf)
    metadata['recommendedModelType'] = recommended_model_type

    return metadata