import os
import sys
import re

import numpy as np

from mst_clustering.cpp_adapters import MstBuilder, DistanceMeasure
from tqdm import tqdm

from time import time as timestamp

def build_mst(data_input):
    if type(data_input) is str:
        dataset = np.load(data_input, allow_pickle=True)
    elif type(data_input) is np.ndarray:
        dataset = data_input
    else:
        raise ValueError("Invalid input to build_mst")

    batch = dataset

    min_vals = dataset.min(axis=0)
    max_vals = dataset.max(axis=0)
    normalized_batch = (batch - min_vals) / (max_vals - min_vals)
    print("Normalized batch of shape %s computed at %f" % (str(normalized_batch.shape), timestamp()))
    builder = MstBuilder(normalized_batch.tolist())
    mst = builder.build(4, DistanceMeasure.EUCLIDEAN)
    print("MSTs built at %f" % timestamp())
    return mst

class SchweinhartIntrinsicDimensionEstimator(object):
    msts: list
    sample_sizes: np.ndarray
    msts_edges_weights: list

    def __init__(self, msts, sample_sizes, e_stats_dict=None, msts_edges_weights=None, save_stats=True):
        self.msts = msts
        self.sample_sizes = sample_sizes
        if e_stats_dict is not None:
            self.e_stats_dict = e_stats_dict
        else:
            self.e_stats_dict = dict()

        self.msts_edges_weights = msts_edges_weights

        self.save_stats = save_stats

    def estimate(self, alpha):
        e_stat = self.get_e_stat(alpha)

        X = np.hstack((np.log(self.sample_sizes)[:, np.newaxis], np.ones((len(self.msts), 1))))
        y = np.log(e_stat)
        w = np.linalg.pinv(X) @ y

        intrinsic_dimension = alpha / (1 - w[0])
        return intrinsic_dimension

    def get_e_stat(self, alpha):
        if alpha in self.e_stats_dict.keys() and self.e_stats_dict[alpha].size == len(self.msts):
            e_stat = self.e_stats_dict[alpha]
        else:
            if self.msts_edges_weights is None:
                self.msts_edges_weights = list()
                for mst in self.msts:
                    self.msts_edges_weights.append(self.get_mst_weights(mst))

            e_stat = np.zeros(len(self.msts))
            for index, mst_weights in enumerate(self.msts_edges_weights):
                e_stat[index] = self.compute_e_stat(mst_weights, alpha)
            if self.save_stats:
                self.e_stats_dict[alpha] = e_stat

        return e_stat

    @staticmethod
    def get_mst_weights(mst):
        edges = np.array(mst.get_tree_edges(*mst.get_roots()))
        extract_weights = np.vectorize(lambda edge: edge.weight)
        weights = np.array(extract_weights(edges))

        return weights

    @staticmethod
    def compute_e_stat(mst_weights, alpha):
        e_stat = np.sum(mst_weights ** alpha)

        return e_stat

def compute_dimensions(data_input, mst, alpha_in=1):
    if type(data_input) is str:
        dataset = np.load(data_input, allow_pickle=True)
    elif type(data_input) is np.ndarray and len(data_input.shape) == 2:
        dataset = data_input
    else:
        raise ValueError("Invalid input to compute_dimensions")
    sizes = np.array([dataset.shape[0]])

    msts_edges_weights = list()
    msts_edges_weights.append(SchweinhartIntrinsicDimensionEstimator.get_mst_weights(mst))

    estimator = SchweinhartIntrinsicDimensionEstimator(
        [mst],
        sizes,
        msts_edges_weights=msts_edges_weights,
        save_stats=False
    )

    if type(alpha_in) is float or type(alpha_in) is int:
        alphas = np.array([alpha_in])
    elif type(alpha_in) is tuple:
        alphas = np.linspace(*alpha_in)
    else:
        raise ValueError("Alpha must be a number or a tuple of three numpy.linspace parameters")
    estimated_dims = np.zeros(alphas.size)

    for index, alpha in tqdm(enumerate(alphas), desc='Computing intrinsic dimensions...'):
        estimated_dims[index] = estimator.estimate(alpha)

    return estimated_dims
