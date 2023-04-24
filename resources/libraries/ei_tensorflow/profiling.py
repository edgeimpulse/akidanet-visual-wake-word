
from __future__ import print_function
import json
import time
import traceback
import os
import numpy as np
import tensorflow as tf
import json, datetime, time, traceback
import os, shutil, operator, functools, time, subprocess, math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss, mean_squared_error
import ei_tensorflow.inference
import ei_tensorflow.brainchip.model
from concurrent.futures import ThreadPoolExecutor
import ei_tensorflow.utils

from ei_tensorflow.constrained_object_detection.util import batch_convert_segmentation_map_to_object_detection_prediction
from ei_tensorflow.constrained_object_detection.metrics import non_background_metrics
from ei_tensorflow.constrained_object_detection.metrics import dataset_match_by_near_centroids

def tflite_predict(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
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

def tflite_predict_object_detection(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
          item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
          item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
          interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
          interpreter.invoke()
          rect_label_scores = ei_tensorflow.inference.process_output_object_detection(output_details, interpreter)
          pred_y.append(rect_label_scores)
          # Print an update at least every 10 seconds
          current_time = time.time()
          if last_log + 10 < current_time:
              print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
              last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolov5(model, version, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            _batch, width, height, _channels = input_details[0]['shape']
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            # expects to have batch dim here, eg (1, 5376, 6)
            # if not, then add batch dim
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            rect_label_scores = ei_tensorflow.inference.process_output_yolov5(output, (width, height),
                version)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolox(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
          item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
          item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
          interpreter.set_tensor(input_details[0]['index'], item_as_tensor)

          _batch, width, height, _channels = input_details[0]['shape']
          if width != height:
            raise Exception(f"expected square input, got {input_details[0]['shape']}")

          interpreter.invoke()
          output = interpreter.get_tensor(output_details[0]['index'])
          # expects to have batch dim here, eg (1, 5376, 6)
          # if not, then add batch dim
          if len(output.shape) == 2:
            output = np.expand_dims(output, axis=0)
          rect_label_scores = ei_tensorflow.inference.process_output_yolox(output, img_size=width)
          pred_y.append(rect_label_scores)
          # Print an update at least every 10 seconds
          current_time = time.time()
          if last_log + 10 < current_time:
              print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
              last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolov7(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
          item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
          item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
          interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
          interpreter.invoke()
          rect_label_scores = ei_tensorflow.inference.process_output_yolov7(output_details,
            width=input_details[0]['shape'][1], height=input_details[0]['shape'][2])
          pred_y.append(rect_label_scores)
          # Print an update at least every 10 seconds
          current_time = time.time()
          if last_log + 10 < current_time:
              print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
              last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_segmentation(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    last_log = time.time()

    y_pred = []
    for item, _ in validation_dataset.take(-1):
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = ei_tensorflow.inference.process_output(output_details, output)
        y_pred.append(output)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(y_pred) - 1))), flush=True)
            last_log = current_time

    y_pred = np.stack(y_pred)

    return y_pred

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

def make_predictions(mode, model, validation_dataset, Y_test, train_dataset,
                            Y_train, test_dataset, Y_real_test, akida_model_path):
    if akida_model_path:
        # TODO: We should avoid vendor-specific naming at this level, for maintainability
        return ei_tensorflow.brainchip.model.make_predictions(mode, akida_model_path, validation_dataset,
                                                              Y_test, train_dataset, Y_train, test_dataset, Y_real_test)

    return make_predictions_tflite(mode, model, validation_dataset, Y_test,
                                   train_dataset, Y_train, test_dataset, Y_real_test)

def make_predictions_tflite(mode, model, validation_dataset, Y_test, train_dataset, Y_train, test_dataset, Y_real_test):
    prediction_train = None
    prediction_test = None

    if mode == 'object-detection':
        prediction = tflite_predict_object_detection(model, validation_dataset, len(Y_test))
    elif mode == 'yolov5':
        prediction = tflite_predict_yolov5(model, 6, validation_dataset, len(Y_test))
    elif mode == 'yolov5v5-drpai':
        prediction = tflite_predict_yolov5(model, 5, validation_dataset, len(Y_test))
    elif mode == 'yolox':
        prediction = tflite_predict_yolox(model, validation_dataset, len(Y_test))
    elif mode == 'yolov7':
        prediction = tflite_predict_yolov7(model, validation_dataset, len(Y_test))
    elif mode == 'segmentation':
        prediction = tflite_predict_segmentation(model, validation_dataset, len(Y_test))
    else:
        prediction = tflite_predict(model, validation_dataset, len(Y_test))
        if (not train_dataset is None) and (not Y_train is None):
            prediction_train = tflite_predict(model, train_dataset, len(Y_train))
        if (not test_dataset is None) and (not Y_real_test is None):
            prediction_test = tflite_predict(model, test_dataset, len(Y_real_test))

    return prediction, prediction_train, prediction_test

def profile_model(model_type, model, model_file, validation_dataset, Y_test, X_samples, Y_samples,
                         has_samples, memory, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                         num_classes, train_dataset=None, Y_train=None, test_dataset=None, Y_real_test=None,
                         akida_model_path=None):
    """Calculates performance statistics for a TensorFlow Lite model"""
    matrix_train=None
    matrix_test=None
    report_train=None
    report_test=None

    prediction, prediction_train, prediction_test = make_predictions(mode, model, validation_dataset, Y_test,
                                                                     train_dataset, Y_train, test_dataset,
                                                                     Y_real_test, akida_model_path)

    if mode == 'classification':
        Y_labels = []
        for ix in range(num_classes):
            Y_labels.append(ix)
        matrix = confusion_matrix(Y_test.argmax(axis=1), prediction.argmax(axis=1), labels=Y_labels)
        report = classification_report(Y_test.argmax(axis=1), prediction.argmax(axis=1), output_dict=True, zero_division=0)
        if not prediction_train is None:
            matrix_train = confusion_matrix(Y_train.argmax(axis=1), prediction_train.argmax(axis=1))
            report_train = classification_report(Y_train.argmax(axis=1), prediction_train.argmax(axis=1), output_dict=True, zero_division=0)
        if not prediction_test is None:
            matrix_test = confusion_matrix(Y_real_test.argmax(axis=1), prediction_test.argmax(axis=1))
            report_test = classification_report(Y_real_test.argmax(axis=1), prediction_test.argmax(axis=1), output_dict=True, zero_division=0)

        accuracy = report['accuracy']
        loss = log_loss(Y_test, prediction)
        try:
            # Make predictions for feature explorer
            if has_samples:
                if model:
                    feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples))
                elif akida_model_path:
                    feature_explorer_predictions = ei_tensorflow.brainchip.model.predict(akida_model_path, X_samples, len(Y_samples))
                else:
                    raise Exception('Expecting either a Keras model or an Akida model')

                # Store each prediction with the original sample for the feature explorer
                prediction_samples = np.concatenate((Y_samples, np.array([feature_explorer_predictions.argmax(axis=1) + 1]).T), axis=1).tolist()
            else:
                prediction_samples = []
        except Exception as e:
            print('Failed to generate feature explorer', e, flush=True)
            prediction_samples = []
    elif mode == 'regression':
        matrix = np.array([])
        report = {}
        accuracy = 0
        loss = mean_squared_error(Y_test, prediction[:,0])
        try:
            # Make predictions for feature explorer
            if has_samples:
                feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples))
                # Store each prediction with the original sample for the feature explorer
                prediction_samples = np.concatenate((Y_samples, feature_explorer_predictions), axis=1).tolist()
            else:
                prediction_samples = []
        except Exception as e:
            print('Failed to generate feature explorer', e, flush=True)
            prediction_samples = []
    elif mode == 'object-detection' or mode == 'yolov5' or mode == 'yolov5v5-drpai' or mode == 'yolox':
        # This is only installed on object detection containers so import it only when used
        from mean_average_precision import MetricBuilder
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)
        # Calculate mean average precision
        def un_onehot(onehot_array):
            """Go from our one-hot encoding to an index"""
            val = np.argmax(onehot_array, axis=0)
            return val
        for index, sample in enumerate(validation_dataset.take(-1).unbatch()):
            data = sample[0]
            labels = sample[1]
            p = prediction[index]
            gt = []
            curr_ps = []

            boxes = labels[0]
            labels = labels[1]
            for box_index, box in enumerate(boxes):
                label = labels[box_index]
                label = un_onehot(label)
                gt.append([box[0], box[1], box[2], box[3], label, 0, 0])

            for p2 in p:
                curr_ps.append([p2[0][0], p2[0][1], p2[0][2], p2[0][3], p2[1], p2[2]])

            gt = np.array(gt)
            curr_ps = np.array(curr_ps)

            metric_fn.add(curr_ps, gt)

        coco_map = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05),
                                   recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        matrix = np.array([])
        report = {}
        accuracy = float(coco_map)
        loss = 0
        prediction_samples = []
    elif mode == 'segmentation':

        _batch, width, height, num_classes = prediction.shape
        if width != height:
            raise Exception("Expected segmentation output to be square, not",
                            prediction.shape)
        output_width_height = width

        # y_true has already been extracted during tflite_predict_segmentation
        # and has labels including implicit background class = 0

        # TODO(mat): what should minimum_confidence_rating be here?
        y_pred = batch_convert_segmentation_map_to_object_detection_prediction(
            prediction, minimum_confidence_rating=0.5, fuse=True)

        # do alignment by centroids. this results in a flatten list of int
        # labels that is suitable for confusion matrix calculations.
        y_true_labels, y_pred_labels = dataset_match_by_near_centroids(
            # batch the data since the function expects it
            validation_dataset.batch(32, drop_remainder=False), y_pred, output_width_height)

        # TODO(mat): we need to pass out recall too for FOMO
        matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=range(num_classes))
        _precision, _recall, f1 = non_background_metrics(y_true_labels, y_pred_labels, num_classes)
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
        accuracy = f1
        loss = 0
        prediction_samples = []

    model_size = 0
    if model:
        model_size = len(model)

    if akida_model_path:
        is_supported_on_mcu = False
        mcu_support_error = "Akida models run only on Linux boards with AKD1000"
    else:
        is_supported_on_mcu, mcu_support_error = check_if_model_runs_on_mcu(model_file, log_messages=False)

    if (is_supported_on_mcu):
        if (not memory):
            memory = calculate_memory(model_file, model_type, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script)
    else:
        memory = {}
        memory['tflite'] = {
            'ram': 0,
            'rom': model_size,
            'arenaSize': 0,
            'modelSize': model_size
        }
        memory['eon'] = {
            'ram': 0,
            'rom': model_size,
            'arenaSize': 0,
            'modelSize': model_size
        }

    return {
        'type': model_type,
        'loss': loss,
        'accuracy': accuracy,
        'accuracyTrain': report_train['accuracy'] if not report_train is None else None,
        'accuracyTest': report_test['accuracy'] if not report_test is None else None,
        'confusionMatrix': matrix.tolist(),
        'confusionMatrixTrain': matrix_train.tolist() if not matrix_train is None else None,
        'confusionMatrixTest': matrix_test.tolist() if not matrix_test is None else None,
        'report': report,
        'reportTrain': report_train,
        'reportTest': report_test,
        'size': model_size,
        'estimatedMACCs': None,
        'memory': memory,
        'predictions': prediction_samples,
        'isSupportedOnMcu': is_supported_on_mcu,
        'mcuSupportError': mcu_support_error,
    }

def run_tasks_in_parallel(tasks, parallel_count):
    res = []
    with ThreadPoolExecutor(parallel_count) as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            res.append(running_task.result())
    return res

def calculate_memory(model_file, model_type, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                     calculate_non_cmsis=False):
    model_size = os.stat(model_file).st_size

    parallel_count = 4
    if (model_size > 1 * 1024 * 1024):
        parallel_count = 1

    # Some models don't have the scripts (e.g. akida) so skip this step
    if prepare_model_tflite_script or prepare_model_tflite_eon_script:
        memory = {}

        def calc_memory(id, title, is_eon, is_non_cmsis):
            try:
                print('Profiling ' + model_type + ' model (' + title + ')...', flush=True)

                benchmark_folder = f'/app/benchmark-{id}'
                script = f'{benchmark_folder}/prepare_tflite_{id}.sh'
                if (is_eon):
                    if (is_non_cmsis):
                        script = f'{benchmark_folder}/prepare_eon_cmsisnn_disabled_{id}.sh'
                    else:
                        script = f'{benchmark_folder}/prepare_eon_{id}.sh'

                out_folder = f'{benchmark_folder}/tflite-model'

                # create prep scripts
                if is_eon:
                    if is_non_cmsis:
                        with open(script, 'w') as f:
                            f.write(prepare_model_tflite_eon_script(model_file, cmsisnn=False, out_folder=out_folder))
                    else:
                        with open(script, 'w') as f:
                            f.write(prepare_model_tflite_eon_script(model_file, cmsisnn=True, out_folder=out_folder))
                else:
                    with open(script, 'w') as f:
                        f.write(prepare_model_tflite_script(model_file, out_folder=out_folder))

                args = [
                    f'{benchmark_folder}/benchmark.sh',
                    '--tflite-type', model_type,
                    '--tflite-file', model_file
                ]
                if is_eon:
                    args.append('--eon')
                if is_non_cmsis:
                    args.append('--disable-cmsis-nn')

                if os.path.exists(f'{benchmark_folder}/tflite-model'):
                    shutil.rmtree(f'{benchmark_folder}/tflite-model')
                subprocess.check_output(['sh', script]).decode("utf-8")
                tflite_output = json.loads(subprocess.check_output(args).decode("utf-8"))
                if os.getenv('K8S_ENVIRONMENT') == 'staging' or os.getenv('K8S_ENVIRONMENT') == 'test':
                    print(tflite_output['logLines'])

                if is_eon:
                    # eon is always correct in memory
                    return { 'id': id, 'output': tflite_output }
                else:
                    # add fudge factor since the target architecture is different
                    # (q: can this go since the changes in https://github.com/edgeimpulse/edgeimpulse/pull/6268)
                    old_arena_size = tflite_output['arenaSize']
                    extra_arena_size = int(math.floor((math.ceil(old_arena_size) * 0.2) + 1024))

                    tflite_output['ram'] = tflite_output['ram'] + extra_arena_size
                    tflite_output['arenaSize'] = tflite_output['arenaSize'] + extra_arena_size

                    return { 'id': id, 'output': tflite_output }
            except Exception as err:
                print('WARN: Failed to get memory (' + title + '): ', end='')
                print(err, flush=True)
                return { 'id': id, 'output': None }

        task_list = []

        if prepare_model_tflite_script:
            task_list.append(lambda: calc_memory(id=1, title='TensorFlow Lite Micro', is_eon=False, is_non_cmsis=False))
            if calculate_non_cmsis:
                task_list.append(lambda: calc_memory(id=2, title='TensorFlow Lite Micro, HW optimizations disabled', is_eon=False, is_non_cmsis=True))
        if prepare_model_tflite_eon_script:
            task_list.append(lambda: calc_memory(id=3, title='EON', is_eon=True, is_non_cmsis=False))
            if calculate_non_cmsis:
                task_list.append(lambda: calc_memory(id=4, title='EON, HW optimizations disabled', is_eon=True, is_non_cmsis=True))

        results = run_tasks_in_parallel(task_list, parallel_count)
        for r in results:
            if (r['id'] == 1):
                memory['tflite'] = r['output']
            elif (r['id'] == 2):
                memory['tflite_cmsis_nn_disabled'] = r['output']
            elif (r['id'] == 3):
                memory['eon'] = r['output']
            elif (r['id'] == 4):
                memory['eon_cmsis_nn_disabled'] = r['output']

    else:
        memory = None

    return memory

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
            print('Error while estimating maccs for layer', flush=True)
            print(err, flush=True)
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

def get_model_metadata(keras_model, validation_dataset, Y_test, X_samples, Y_samples, has_samples,
                       class_names, object_detection_last_layer, curr_metadata, mode, prepare_model_tflite_script,
                       prepare_model_tflite_eon_script, model_float32=None, model_int8=None,
                       file_float32=None, file_int8=None, file_akida=None,
                       train_dataset=None, Y_train=None, test_dataset=None, Y_real_test=None):

    metadata = {
        'metadataVersion': 5,
        'created': datetime.datetime.now().isoformat(),
        'classNames': class_names,
        'availableModelTypes': [],
        'recommendedModelType': '',
        'modelValidationMetrics': [],
        'modelIODetails': [],
        'mode': mode,
        'kerasJSON': None,
        'performance': None,
        'objectDetectionLastLayer': object_detection_last_layer
    }

    recalculate_memory = True
    recalculate_performance = True

    # e.g. ONNX conversion failed
    if (file_int8 and not os.path.exists(file_int8)):
        file_int8 = None

    # For some model types (e.g. object detection) there is no keras model, so
    # we are unable to compute some of our stats with these methods
    if keras_model:
        # This describes the basic inputs and outputs, but skips over complex parts
        # such as transfer learning base models
        metadata['layers'] = describe_layers(keras_model)
        estimated_maccs = estimate_maccs_for_model(keras_model)
        # This describes the full model, so use it to determine if the architecture
        # has changed between runs
        metadata['kerasJSON'] = keras_model.to_json()
        # Only recalculate memory when model architecture has changed
        if (
            curr_metadata and 'kerasJSON' in curr_metadata and 'metadataVersion' in curr_metadata
            and curr_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_metadata['kerasJSON']
        ):
            recalculate_memory = False
        else:
            recalculate_memory = True

        if (
            curr_metadata and 'kerasJSON' in curr_metadata and 'metadataVersion' in curr_metadata
            and curr_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_metadata['kerasJSON']
            and 'performance' in curr_metadata
            and curr_metadata['performance']
        ):
            metadata['performance'] = curr_metadata['performance']
            recalculate_performance = False
        else:
            recalculate_memory = True
            recalculate_performance = True
    else:
        metadata['layers'] = []
        estimated_maccs = -1
        # If there's no Keras model we can't tell if the architecture has changed, so recalculate memory every time
        recalculate_memory = True
        recalculate_performance = True

    if recalculate_performance:
        try:
            args = '/app/profiler/build/profiling '
            if file_float32:
                args = args + file_float32 + ' '
            if file_int8:
                args = args + file_int8 + ' '

            print('Calculating inferencing time...', flush=True)
            a = os.popen(args).read()
            if '{' in a and '}' in a:
                metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
                print('Calculating inferencing time OK', flush=True)
            else:
                print('Failed to calculate inferencing time:', a)
        except Exception as err:
            print('Error while calculating inferencing time:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            metadata['performance'] = None

    float32_perf = None
    int8_perf = None

    if model_float32:
        try:
            print('Profiling float32 model...', flush=True)
            model_type = 'float32'

            memory = None
            if not recalculate_memory:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            float32_perf = profile_model(model_type, model_float32, file_float32, validation_dataset, Y_test, X_samples, Y_samples, has_samples, memory, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script, len(class_names), train_dataset, Y_train, test_dataset, Y_real_test)
            float32_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(float32_perf)
            metadata['modelIODetails'].append(get_io_details(model_float32, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite float32 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if model_int8:
        try:
            print('Profiling int8 model...', flush=True)
            model_type = 'int8'

            memory = None
            if not recalculate_memory:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            int8_perf = profile_model(model_type, model_int8, file_int8, validation_dataset, Y_test, X_samples, Y_samples, has_samples, memory, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script, len(class_names), train_dataset, Y_train, test_dataset, Y_real_test)
            int8_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(int8_perf)
            metadata['modelIODetails'].append(get_io_details(model_int8, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite int8 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if file_akida:
        print('Profiling akida model...', flush=True)
        model_type = 'akida'

        program_size, total_nps, nodes = ei_tensorflow.brainchip.model.get_hardware_utilization(file_akida)
        flops, macs = ei_tensorflow.brainchip.model.get_macs_flops(keras_model)
        memory = {}
        memory['tflite'] = {
            'ram': -1,
            'rom': program_size,
            'arenaSize': 0,
            'modelSize': 0
        }
        # only 'eon' is used, see comment in populateMetadataTemplate in
        # studio/client/project/pages/training-keras-ui.ts
        memory['eon'] = {
            'ram': -1,
            'rom': program_size,
            'arenaSize': 0,
            'modelSize': 0
        }
        akida_perf = profile_model(model_type, None, None, validation_dataset, Y_test, X_samples,
                                   Y_samples, has_samples, memory, mode, None,
                                   None, len(class_names), train_dataset,
                                   Y_train, test_dataset, Y_real_test, file_akida)
        sparsity = ei_tensorflow.brainchip.model.get_model_sparsity(file_akida, mode, validation_dataset)
        print("########################################")
        print(f"Model sparsity: {sparsity:6.2f} %")
        print(f"Used NPs:   {total_nps:10d}")
        print(f"Used nodes: {nodes:10d}")
        print(f"FLOPS: {flops:15d}")
        print(f"MACs: {macs:20.8}")
        print("########################################")
        akida_perf['estimatedMACCs'] = macs
        metadata['availableModelTypes'].append(model_type)
        metadata['modelValidationMetrics'].append(akida_perf)

    # Decide which model to recommend
    if file_akida:
        metadata['recommendedModelType'] = 'akida'
    else:
        recommended_model_type = get_recommended_model_type(float32_perf, int8_perf)
        metadata['recommendedModelType'] = recommended_model_type

    return metadata

def profile_tflite_file(file, model_type, mode,
                        prepare_model_tflite_script,
                        prepare_model_tflite_eon_script):
    metadata = {
        'tfliteFileSizeBytes': os.path.getsize(file)
    }
    try:
        args = '/app/profiler/build/profiling ' + file

        print('Calculating inferencing time...', flush=True)
        a = os.popen(args).read()
        metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
        print('Calculating inferencing time OK', flush=True)
    except Exception as err:
        print('Error while calculating inferencing time:', flush=True)
        print(err, flush=True)
        traceback.print_exc()
        metadata['performance'] = None


    is_supported_on_mcu, mcu_support_error = check_if_model_runs_on_mcu(file, log_messages=True)
    metadata['isSupportedOnMcu'] = is_supported_on_mcu
    metadata['mcuSupportError'] = mcu_support_error

    if (metadata['isSupportedOnMcu']):
        metadata['memory'] = calculate_memory(file, model_type, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                                              calculate_non_cmsis=True)
    return metadata

def check_if_model_runs_on_mcu(file, log_messages):
    is_supported_on_mcu = True
    mcu_support_error = None

    try:
        if log_messages:
            print('Determining whether this model runs on MCU...')
        result = subprocess.run(['/app/eon_compiler/compiler', '--verify', file], stdout=subprocess.PIPE)
        if (result.returncode == 0):
            stdout = result.stdout.decode('utf-8')
            msg = json.loads(stdout)

            arena_size = msg['arena_size']
            # more than 6MB
            if arena_size > 6 * 1024 * 1024:
                is_supported_on_mcu = False
                mcu_support_error = 'Calculated arena size is >6MB'
            else:
                is_supported_on_mcu = True
        else:
            is_supported_on_mcu = False
            stdout = result.stdout.decode('utf-8')
            if stdout != '':
                mcu_support_error = stdout
            else:
                mcu_support_error = 'Verifying model failed with code ' + str(result.returncode) + ' and no error message'
        if log_messages:
            print('Determining whether this model runs on MCU OK')
    except Exception as err:
        print('Determining whether this model runs on MCU failed:', flush=True)
        print(err, flush=True)
        traceback.print_exc()
        is_supported_on_mcu = False
        mcu_support_error = str(err)

    return is_supported_on_mcu, mcu_support_error
