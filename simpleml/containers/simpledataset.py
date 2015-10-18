import numpy as np


class SimpleDataset:

    ALLOWED_SCALE_TYPES = ['unit']

    def __init__(self, ds, label_idx=-1):
        self.ds = ds
        self.label_idx = label_idx

        # Split dataset into features/labels
        if label_idx == -1:
            self._features = ds[:, :-1]
        else:
            self._features = np.hstack([ds[:, :label_idx], ds[:, label_idx+1:]])
        self._labels = ds[:, label_idx]

        # Any potential scaled or expanded features
        self.operating_features = np.zeros_like(self._features, dtype=np.float64)
        self.scaled = False
        self.expanded_basis = False
        self.expanded_features = []
        self.expand_funcs = []

    def scale_features(self, scale_type='unit'):
        if scale_type not in ['unit']:
            raise ValueError('scale_type must be one of {}, got {}'.format(ALLOWED_SCALE_TYPES, scale_type))

        scalevals = 1 / np.max(np.abs(self.operating_features), axis=0)
        if not self.operating_features.all():
            np.copyto(self.operating_features, self._features)
        self.operating_features *= scalevals
        self.scaled = True
        return scalevals

    def expand_basis(self, *args):
        args = [lambda x: x] + list(args)
        for a in args:
            if not callable(a):
                raise ValueError('all args must be callable, got {}'.format(a))

        self.expand_funcs = args
        if not self.operating_features.all():
            np.copyto(self.operating_features, self._features)
        self.operating_features = np.hstack([f(self.operating_features) for f in args])
        self.expended_basis = True

    def features(self):
        if not self.operating_features.all():
            return self._features
        return self.operating_features

    def labels(self):
        return self._labels

    def raw_features(self):
        return self._features

    def raw_data(self):
        return self.ds

    def add_bias(self):
        if not self.operating_features.all():
            np.copyto(self.operating_features, self._features)
        ones = np.ones([len(self.operating_features), 1])
        self.operating_features = np.hstack([self.operating_features, ones])
