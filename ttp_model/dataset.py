# coding=utf-8
"""
Graph - the name of the current project.
instrument.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
6 / 15 / 18 - the current system date.
8: 03 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

import numpy
import datetime
import collections
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


Datasets = collections.namedtuple('Datasets', ['train', 'test'])


class Labels:
    """
    Label deffinitions
    """
    BUY = 1
    HOLD = 0
    SELL = 0


class DataSet:
    """
    Container class for a dataset (deprecated).
    """
    
    def __init__(self, features, labels, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None, hm_classes=None):
        """
        Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        assert features.shape[0] == labels.shape[0], (
          'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
        classes = list(set(labels))
        hm_classes = hm_classes or int(max(classes)) + 1
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert features.shape[3] == 1
        features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
        if dtype == dtypes.float32:
            features = features.astype(numpy.float32)
        self._features = features
        self._set_labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if one_hot:
            hot_array = numpy.zeros((self.num_examples, hm_classes))
            hot_array[numpy.arange(self.num_examples), labels] = 1
            self._set_labels = hot_array
    
    @property
    def features(self):
        """
        getter method for features
        :return: features
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        setter method for features
        :param value: new feature
        :return: appended features
        """
        self._features = numpy.append(self._features, value)

    @property
    def labels(self):
        """
        getter method for labels
        :return: labels
        """
        return self._set_labels

    @labels.setter
    def labels(self, value):
        """
        setter method for labels
        :param value: new labels
        :return: appended labels
        """
        self._set_labels = numpy.append(self._set_labels, value)

    @property
    def num_examples(self):
        """
        getter method for number of datasets
        :return:
        """
        return self.features.shape[0]
    
    @property
    def epochs_completed(self):
        """
        getter method for the number of completed epochs
        :return:
        """
        return self._epochs_completed

    def append(self, obj):
        """
        merge to datasets
        :param obj: the dataset
        :return:
        """
        self.features = obj.features
        self.labels = obj.labels

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self.num_examples)
            numpy.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._set_labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            features_rest_part = self._features[start:self.num_examples]
            labels_rest_part = self._set_labels[start:self.num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self.num_examples)
                numpy.random.shuffle(perm)
                self._features = self.features[perm]
                self._set_labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._set_labels[start:end]
            return numpy.concatenate(
              (features_rest_part, features_new_part)), numpy.concatenate(
                  (labels_rest_part, labels_new_part))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._set_labels[start:end]
