import sys, os, json
import numpy as np
sys.path.append('./resources/libraries')
from ei_sklearn.gmm_anomaly_detection import GaussianMixtureAnomalyScorer

dir_path = os.path.dirname(os.path.realpath(__file__))
model_filename = "model_tflite32"
input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(os.path.join(dir_path, 'options.json')) as f:
    options = json.load(f)

GaussianMixtureAnomalyScorer.run_classify_job(dir_path, model_filename,
    input_filename, options['axes'], output_filename)