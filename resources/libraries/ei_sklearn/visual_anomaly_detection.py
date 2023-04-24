from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .translate import translate_function

import numpy as np
from jax import lax, vmap, jit
import jax.numpy as jnp

def _mobile_net_trunk_imagenet_96_weights(num_channels: int):
    # note: regardless of what resolution we intend to use for actual image
    #  input we emperically get the best result for anomaly detection from
    #  using 96x96 imagenet weights. i (mat) suspect this is due to fact the
    #  anomaly detection features are usually not large, so the lower the
    #  resolution of the pretrained weights the better.

    #TODO: refactor weight downloading between this and ei_tensorflow.constrained_object_detection.training
    if num_channels == 1:
        weights = "./transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.96x96.grayscale.bsize_64.lr_0_005.epoch_260.val_loss_3.10.val_accuracy_0.35.hdf5"
    elif num_channels == 3:
        weights = "./transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5"
    else:
        raise Exception("Pretrained weights only available for num_channels 1 or 3")

    mobile_net_v2 = MobileNetV2(input_shape=(96, 96, num_channels),
                                weights=weights,
                                alpha=0.35, include_top=True)
    cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
    mobile_net_trunk = Model(inputs=mobile_net_v2.input, outputs=cut_point.output)
    return mobile_net_trunk.get_weights()

class MobileNetFeatureExtractor(object):

    def __init__(self,
                 input_shape: Tuple[int],
                 use_mobile_net_pretrained_weights: bool,
                 seed: int):
        """ Mobile Net Feature Extractor
        Args:
            input_shape: (H,W,C) shape of expected input. Used to build
                MobileNet trunk.
            use_mobile_net_pretrained_weights: if true initialise MobileNet
                with ImageNet weights for 96x96 input. We use 96x96 weights
                since we'll only being used the start of mobilenet to reduce
                to 1/8th input.
        """

        # validate input shape
        if len(input_shape) != 3:
            raise Exception(f"Expected input_shape be (H,W,C), not {input_shape}")
        _img_height, _img_width, num_channels = input_shape
        if num_channels not in [1, 3]:
            raise Exception(f"VisualAD only supports num_channels of 1 or 3,"
                            f" not {num_channels} from input_shape {input_shape}")
        self.input_shape = input_shape

        # construct mobile net trunk
        if seed is not None:
            tf.random.set_seed(seed)
        mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                    weights=None, alpha=0.35, include_top=False)
        cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
        self.mobile_net_trunk = Model(inputs=mobile_net_v2.input,
                                      outputs=cut_point.output)

        # only load in 96x96 imagenet weights if requested
        if use_mobile_net_pretrained_weights:
            self.mobile_net_trunk.set_weights(
                _mobile_net_trunk_imagenet_96_weights(num_channels))

    def extract_features(self, x: np.array):
        _batch, img_height, img_width, num_channels = x.shape
        if self.input_shape != (img_height, img_width, num_channels):
            raise Exception(f"Expected input to be batched {self.input_shape}"
                            f" not {x.shape}")
        return self._batch_run(x)

    def _batch_run(self, x: np.array, batch_size: int=64):
        # TODO(mat) will only have to do these during training, not inference
        if len(x) < batch_size:
            return self.mobile_net_trunk(x).numpy()
        idx = 0
        y = []
        while idx < len(x):
            x_batch = x[idx:idx+batch_size]
            y.append(self.mobile_net_trunk(x_batch).numpy())
            idx += batch_size
        return np.concatenate(y)


class SpatialAwareRandomProjection(object):

    def __init__(self,
                 random_projection_dim: int,
                 seed: int):
        self.random_projection_dim = random_projection_dim
        self.seed = seed
        self.fit_and_project_called = False

    def fit_and_project(self, x: np.array):
        # record details of the shapes of x; specifically
        # the spatial component of the shape (i.e. everything but the last
        # dimension). we do this since we're going to have to flatten x
        # to be able to run through the sklearn projection which only
        # supports 2D data.
        spatial_shape = x.shape[:-1]
        x_dimension = x.shape[-1]

        # convert from, say, (num_instances=10, height=3, width=3,
        # n_features=96) to flattened (90, 96)
        flat_x = x.reshape((-1, x_dimension))

        # "fit" the projection (which, for this, just creates the
        # projection matrix)
        self.random_projection = GaussianRandomProjection(
                n_components=self.random_projection_dim,
                random_state=self.seed)
        self.random_projection.fit(flat_x)

        # apply the projection which will go from, say, (90, 96) -> (90, 8) if
        # random_projection_dim=8
        flat_x = self.random_projection.transform(flat_x).astype(np.float32)

        # restore the original spatial shape; e.g. (90, 8) -> (10, 3, 3, 8)
        projected_x = flat_x.reshape((*spatial_shape, self.random_projection_dim))
        self.fit_and_project_called = True
        return projected_x

    def project(self, y: np.array, use_jax: bool=False):
        if not self.fit_and_project_called:
            raise Exception("Must call fit_and_project() before project()")
        if use_jax:
            project_fn = translate_function(
                self.random_projection, GaussianRandomProjection.transform)
            spatial_project_fn = vmap(vmap(project_fn))
            return spatial_project_fn(y)
        else:
            spatial_shape = y.shape[:-1]
            y_dimension = y.shape[-1]
            flat_y = y.reshape((-1, y_dimension))
            flat_y = self.random_projection.transform(flat_y).astype(np.float32)
            return flat_y.reshape((*spatial_shape, self.random_projection_dim))


class AveragePooling(object):
    # minimal port of dh-haiku avg pool (so we don't need to pull in the
    # entire package just for this one piece of code)
    # see https://github.com/deepmind/dm-haiku/blob/ab16af8230b1be279cf99a660e0fe95bd759e977/haiku/_src/pool.py#L105
    # assumes x in 4d... TODO(mat) should we pull in _infer_shape too?

    def __init__(self, pool_size: int, pool_stride: int):
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def __call__(self, x: np.array):
        window_shape = (1, self.pool_size, self.pool_size, 1)
        strides = (1, self.pool_stride, self.pool_stride, 1)
        padding = 'VALID'
        reduce_window_args = (0., lax.add, window_shape, strides, padding)
        pooled = lax.reduce_window(x, *reduce_window_args)
        return pooled / np.prod(window_shape)

class SpatialAwareGaussianMixtureAnomalyScorer(object):

    def __init__(self, n_components: int, seed: int):
        self.gmm = GaussianMixture(
            n_components=n_components, random_state=seed,
            covariance_type='full')
        self.scaler = StandardScaler()
        self.fit_called = False

    def fit(self, x: np.array):
        # TODO(mat): pull this out into util, share with random projection
        # flat for GMM and scalar
        x_dimension = x.shape[-1]
        flat_x = x.reshape((-1, x_dimension))
        # fit GMM and score
        self.gmm.fit(flat_x)
        scores = self.gmm.score_samples(flat_x)
        # use scores to fit scalar
        # note: scalar requires trailing dimension
        scores = np.expand_dims(scores, axis=-1)
        self.scaler.fit(scores)
        self.fit_called = True

    def anomaly_score(self, x: np.array, use_jax: bool=False):
        if not self.fit_called:
            raise Exception("Must call fit() before anomaly_score()")

        if use_jax:
            # for the jax versions we can compose a scalar version function as
            # gmm_score -> standardise -> absolute and then create a spatial
            # version with two vmaps.

            # convert inference functions
            gmm_score_fn = translate_function(
                self.gmm, GaussianMixture.score_samples)
            standardise_fn = translate_function(
                self.scaler, StandardScaler.transform)
            # stitch them together into one function (with absolute)
            def single_element_score_fn(x):
                scores = gmm_score_fn(x)
                scores = standardise_fn(scores)
                return jnp.abs(scores)
            # compile vectorised form for spatial version and return
            spatial_score_fn = vmap(vmap(single_element_score_fn))
            return spatial_score_fn(x)

        else:
            # for the non jax version we need to flatten the x before running
            # the gmm_score -> standardise -> absolute before restoring the
            # shape with a reshape.

            # flatten
            spatial_shape = x.shape[:-1]
            x_dimension = x.shape[-1]
            flat_x = x.reshape((-1, x_dimension))
            # score via GMM
            scores = self.gmm.score_samples(flat_x)
            # standardise with absolute value
            scores = np.expand_dims(scores, axis=-1)
            scores = self.scaler.transform(scores)
            scores = np.abs(scores)
            # return with restored spatial shape
            return scores.reshape(spatial_shape)


class VisualAnomalyDetection(object):

    def __init__(self,
                 input_shape: Tuple[int],
                 use_mobile_net_pretrained_weights: bool,
                 random_projection_dim: int,
                 pool_size: int,
                 pool_stride: int,
                 gmm_n_components: int,
                 seed: int):
        """ Visual Anomaly Detection.
            input_shape: (H,W,C) shape of expected input. Used to build
                MobileNet trunk. see MobileNetFeatureExtractor.
            use_mobile_net_pretrained_weights: if true initialise MobileNet
                with ImageNet weights for 96x96 input. We use 96x96 weights
                since we'll only being used the start of mobilenet to reduce
                to 1/8th input. see MobileNetFeatureExtractor.
            random_projection_dim: projection dimension for spatially aware
                random projection to run on feature maps from mobilenet. if
                None no random projection is used.
            pool_size: pooling kernel size (square) for average pooling post
                random projection.
            pool_stride: pooling stride for average pooling post random
                projection.
            gmm_n_components: num components to pass to spatially aware mixture
                model for scoring
            seed: seed for random number generation.
        """
        self.input_shape = input_shape
        self.feature_extractor = MobileNetFeatureExtractor(
            input_shape, use_mobile_net_pretrained_weights, seed)
        self.feature_map_shape = None
        if random_projection_dim is not None:
            self.random_projection = SpatialAwareRandomProjection(
                random_projection_dim, seed)
        else:
            self.random_projection = None
        self.avg_pooling = AveragePooling(pool_size, pool_stride)
        self.mixture_model = SpatialAwareGaussianMixtureAnomalyScorer(
            gmm_n_components, seed)

    def fit(self, x: np.array):
        feature_map = self.feature_extractor.extract_features(x)
        self.feature_map_shape = feature_map.shape[1:]
        if self.random_projection is not None:
            feature_map = self.random_projection.fit_and_project(feature_map)
        pooled_feature_map = self.avg_pooling(feature_map)
        self.mixture_model.fit(pooled_feature_map)

    def feature_extractor_fn(self):
        return self.feature_extractor.extract_features

    def feature_extractor_input_shape(self):
        return self.input_shape

    def spatial_anomaly_score_fn(self,
                                 reduction_mode: str=None,
                                 use_jax: bool=False):
        def score_fn(feature_map):
            if self.random_projection is not None:
                feature_map = self.random_projection.project(
                    feature_map, use_jax=use_jax)
            pooled_feature_map = self.avg_pooling(feature_map)
            spatial_scores = self.mixture_model.anomaly_score(
                pooled_feature_map, use_jax=use_jax)
            if reduction_mode is None:
                return spatial_scores
            elif reduction_mode == 'mean':
                return spatial_scores.mean(axis=(-1, -2))
            elif reduction_mode == 'max':
                return spatial_scores.max(axis=(-1, -2))
            else:
                raise Exception(f"Invalid reduction_mode [{reduction_mode}], expected [None, mean, max]")

        # if using jax, jit compile here
        if use_jax:
            score_fn = jit(score_fn)

        return score_fn

    def spatial_anomaly_score_input_shape(self):
        if self.feature_map_shape is None:
            raise Exception("The output shape of the feature extractor, and"
                            " hence the input shape to the spatial anomaly"
                            " scoring, is unknown until .fit() called.")
        return self.feature_map_shape

    def score(self, x: np.array,
                reduction_mode: str=None,
                use_jax: bool=False,
                batch_size: int=64):

        feature_extraction_fn = self.feature_extractor_fn()

        spatial_anomaly_score_fn = self.spatial_anomaly_score_fn(
            reduction_mode, use_jax
        )

        # for very large x, e.g. benchmarking, we need to batch the score_fn
        if batch_size is None:
            feature_map = feature_extraction_fn(x)
            scores = spatial_anomaly_score_fn(feature_map)
            return np.array(scores)
        else:
            idx = 0
            scores = []
            n_batches = 0
            while idx < len(x):
                x_batch = x[idx:idx+batch_size]
                feature_map = feature_extraction_fn(x_batch)
                batch_scores = spatial_anomaly_score_fn(feature_map)
                scores.append(np.array(batch_scores))
                idx += batch_size
                n_batches += 1
            return np.concatenate(scores)
