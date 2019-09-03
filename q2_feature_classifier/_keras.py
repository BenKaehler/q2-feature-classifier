# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json

from q2_types.feature_data import (DNAIterator, DNAFASTAFormat)
import pandas as pd

from .plugin_setup import plugin, citations


class DNAEncoder(object):
    _encoding = {'A' : [1,0,0,0],
                'C' : [0,1,0,0],
                'G' : [0,0,1,0],
                'T' : [0,0,0,1],
                'N' : [0.25,0.25,0.25,0.25],
                'X' : [0.25,0.25,0.25,0.25],
                'K' : [0,0,0.5,0.5],
                'M' : [0.5,0.5,0,0],
                'R' : [0.5,0,0.5,0],
                'S' : [0,0.5,0.5,0],
                'W' : [0.5,0,0,0.5],
                'Y' : [0,0.5,0,0.5],
                'H' : [1/3,1/3,0,1/3],
                'D' : [1/3,0,1/3,1/3],
                'V' : [1/3,1/3,1/3,0],
                'B' : [0,1/3,1/3,1/3],
                '-' : [0,0,0,0],
                '.' : [0,0,0,0]}
    
    def __init__(self, pad_length=None):
        self.pad_length = pad_length
        
    def fit(self, X):
        self.pad_length = max(len(s) for s in X)
    
    def transform(self, X):
        def transform_one(seq):
            x = [self._encoding[c] for c in seq[:self.pad_length]]
            x += [[0,0,0,0]]*max(0, self.pad_length - len(x))
            return array(x).reshape(self.pad_length, 4, -1)
        
        return array([transform_one(seq) for seq in X])


class TaxonomicGenerator(Sequence):
    def __init__(self, X, y, X_encoder, y_encoder, batch_size):
        self.X = X
        self.y = y
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.X) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = X_encoder.transform(batch_X)
        
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = y_encoder.transform(batch_y)
                
        return batch_X, batch_y


def fit_classifier_keras(reference_reads: DNAIterator,
                         reference_taxonomy: pd.Series,
                         classifier: KerasClassifier,
                         class_weight: biom.Table=None,
                         batch_size: int=256,
                         epochs: int=50) -> KerasClassifier:
    classifier.x_encoder.fit(reference_reads)
    classifier.y_encoder.fit(reference_taxonomy)
    generator = TaxonomicGenerator(reference_reads, reference_taxonomy,
                                   classifier.x_encoder, classifier.y_encoder,
                                   batch_size)
    classifier.model.fit_generator(generator, epochs=epochs)
    return model


def classify_keras(reads: DNAFASTAFormat, classifier: KerasClassifier,
                   confidence: float, read_orientation: str = None
                   ) -> pd.DataFrame:
    X = classifier.x_encoder(reads)

    result = None
    return result
                         


