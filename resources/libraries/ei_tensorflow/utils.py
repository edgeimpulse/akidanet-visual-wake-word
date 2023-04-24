import os, json
import tensorflow as tf
import numpy as np

def is_y_structured(file_path):
    with open(file_path, 'rb') as f:
        first_byte = f.read(1)
        if (first_byte == b'{'):
            return True
        else:
            return False

def load_y_structured(dir_path, file_name, num_samples):
    with open(os.path.join(dir_path, file_name), 'r') as file:
        Y_structured_file = json.loads(file.read())
    if not Y_structured_file['version'] or Y_structured_file['version'] != 1:
        print('Unknown version for structured labels. Cannot continue, please contact support.')
        exit(1)

    Y_structured = Y_structured_file['samples']

    if len(Y_structured) != num_samples:
        print('Structured labels should have same length as samples array. Cannot continue, please contact support.')
        exit(1)

    return Y_structured

def load_validation_split_metadata(dir_path, file_name):
    validation_split_metadata = None
    validation_split_metadata_path = os.path.join(dir_path, file_name)
    if (not os.path.exists(validation_split_metadata_path)):
        return None

    with open(validation_split_metadata_path, 'r') as file:
        return json.loads(file.read())

def convert_box_coords(box: dict, width_height: int):
    # TF standard format is [y_min, x_min, y_max, x_max]
    # expressed from 0 to 1
    return [box['y'] / width_height,
            box['x'] / width_height,
            (box['y'] + box['h']) / width_height,
            (box['x'] + box['w']) / width_height]

def process_bounding_boxes(raw_boxes: list, width_height: int, num_classes: int):
    boxes = []
    classes = []
    for box in raw_boxes:
        coords = convert_box_coords(box, width_height)
        boxes.append(coords)
        # The model expects classes starting from 0
        # TODO: Use a more efficient way of doing one hot
        classes.append(tf.one_hot(box['label'] - 1, num_classes).numpy())

    # We have to make sure the correct shape is propagated even for lists that have zero elements
    boxes_tensor = tf.ragged.constant(boxes, inner_shape=[len(raw_boxes), 4])
    classes_tensor = tf.ragged.constant(classes, inner_shape=[len(raw_boxes), num_classes])
    return tf.ragged.stack([boxes_tensor, classes_tensor], axis=0)

def calculate_freq(interval):
    """Determines the frequency of a signal given its interval

    Args:
        interval (_type_): Interval in ms

    Returns:
        _type_: Frequency in Hz
    """
    # Determines the frequency of a signal given its interval
    freq = 1000 / interval
    if abs(freq - round(freq)) < 0.001:
        freq = round(freq)
    return freq
