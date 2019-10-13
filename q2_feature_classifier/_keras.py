# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json

from q2_types.feature_data import DNAIterator
import pandas as pd
import gensim
from sklearn.preprocessing import OneHotEncoder

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
        if not self.pad_length:
            self.pad_length = max(len(s) for s in X)
    
    def transform(self, X):
        def transform_one(seq):
            x = [self._encoding[c] for c in seq[:self.pad_length]]
            x += [[0,0,0,0]]*max(0, self.pad_length - len(x))
            return array(x).reshape(self.pad_length, 4, -1)
        
        return array([transform_one(seq) for seq in X])


class Seq2VecEncoder(object):
    def __init__(self, k=7, size=300, window=5, pad_length=None, workers=1):
        self.k = k
        self.size = size
        self.window = window
        self.pad_length = pad_length
        self.workers = workers

    def fit(self, X):
        seqs = [[s[i:i+self.k] for i in range(len(s)-self.k+1)] for s in seqs]
        if not self.pad_length:
            self.pad_length = max(len(s) for s in seqs)
        model = gensim.models.Word2Vec(
                    sentences=seqs, size=self.size, window=self.window,
                    workers=self.workers, min_count=1)
        self.weights = model.wv

    def transform(self, X):
        def transform_one(seq):
            sentence = zeros((self.pad_length, self.size))
            for i in range(len(seq)- self.k+1):
                sentence[i] = self.weights[s._string[i:i+self.k]]
            return sentence

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
                         classifier_specification: str,
                         sequence_encoder: str='Seq2VecEncoder',
                         read_length: int=300,
                         k: int=7,
                         vec_length: int=300,
                         window: int=5,
                         n_jobs: int=1,
                         loss: str='categorical_crossentropy',
                         optimizer: str='adam',
                         batch_size: int=256,
                         epochs: int=50) -> tuple:
    X, y  = zip((str(s), [reference_taxonomy[s.metadata['id']]])
                for s in reference_reads)

    if sequence_encoder == 'DNAEncoder':
        x_encoder = DNAEncoder(read_length)
    elif sequence_encoder == 'Seq2VecEncoder':
        x_encoder = Seq2VecEncoder(k, vec_length, window, read_length, n_jobs)
    elif sequence_encoder == 'KmerEncoder':
        raise NotImplementedError()

    y_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    x_encoder.fit(X)
    y_encoder.fit(y)

    generator = TaxonomicGenerator(X, y, x_encoder, y_encoder, batch_size)

    model = model_from_json(classifier_specification)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit_generator(generator, epochs=epochs)

    return x_encoder, y_encoder, model


def classify_keras(reads: DNAIterator, classifier: tuple,
                   confidence: float, batch_size: int=256
                   ) -> pd.DataFrame:
    x_encoder, y_encoder, model = classifier
    X, seq_ids  = zip((str(s), s.metadata['id']) for s in reads)
    generator = TaxonomicGenerator(X, [[]]*len(X), x_encoder, y_encoder,
                                   batch_size)
    y = model.predict_generator(X)
    if confidence < 0:
        y = y_encoder.inverse_transform(y)
        confidence = [-1]*len(y)
    else:
        raise NotImplementedError()
    
    result = pd.DataFrame(dict(Taxon=taxonomy, Confidence=confidence)
                          index=seq_ids, columns=['Taxon', 'Confidence'])
    result.index.name = 'Feature ID'
    return result
