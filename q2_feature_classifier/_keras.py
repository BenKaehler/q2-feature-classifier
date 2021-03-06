# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json

from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAIterator)
from q2_types.feature_table import FeatureTable, RelativeFrequency
from qiime2.plugin import Int, Str, Float, Choices, Range
import pandas as pd
from numpy import array, zeros, ceil
import gensim
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from keras.utils import Sequence as KerasSequence
from keras.models import model_from_json
import biom

from .plugin_setup import plugin  # , citations
from ._keras_classifier import (
    ClassifierSpecification, KerasClassifier, Klassifier)
from .classifier import _load_class
from ._consensus_assignment import _get_default_unassignable_label


class DNAEncoder(BaseEstimator):
    _encoding = {'A': [1, 0, 0, 0],
                 'C': [0, 1, 0, 0],
                 'G': [0, 0, 1, 0],
                 'T': [0, 0, 0, 1],
                 'N': [0.25, 0.25, 0.25, 0.25],
                 'X': [0.25, 0.25, 0.25, 0.25],
                 'K': [0, 0, 0.5, 0.5],
                 'M': [0.5, 0.5, 0, 0],
                 'R': [0.5, 0, 0.5, 0],
                 'S': [0, 0.5, 0.5, 0],
                 'W': [0.5, 0, 0, 0.5],
                 'Y': [0, 0.5, 0, 0.5],
                 'H': [1/3, 1/3, 0, 1/3],
                 'D': [1/3, 0, 1/3, 1/3],
                 'V': [1/3, 1/3, 1/3, 0],
                 'B': [0, 1/3, 1/3, 1/3],
                 '-': [0, 0, 0, 0],
                 '.': [0, 0, 0, 0]}

    def __init__(self, pad_length=None):
        self.pad_length = pad_length

    def fit(self, X):
        if not self.pad_length:
            self.pad_length = max(len(s) for s in X)

    def transform(self, X):
        def transform_one(seq):
            x = [self._encoding[c] for c in seq[:self.pad_length]]
            x += [[0, 0, 0, 0]]*max(0, self.pad_length - len(x))
            return array(x)  # .reshape(self.pad_length, 4, -1)

        return array([transform_one(seq) for seq in X])

    def get_params(self):
        return dict(pad_length=self.pad_length)


class Seq2VecEncoder(BaseEstimator):
    def __init__(self, k=7, size=300, window=5, pad_length=None, workers=1,
                 weights=None):
        self.k = k
        self.size = size
        self.window = window
        self.pad_length = pad_length
        self.workers = workers
        self.weights = weights

    def fit(self, X):
        seqs = [[s[i:i+self.k] for i in range(len(s)-self.k+1)] for s in X]
        if not self.pad_length:
            self.pad_length = max(len(s) for s in seqs)
        model = gensim.models.Word2Vec(
                    sentences=seqs, size=self.size, window=self.window,
                    workers=self.workers, min_count=1, sg=1)
        self.weights = model.wv

    def transform(self, X):
        def transform_one(seq):
            sentence = zeros((self.pad_length, self.size))
            for i in range(min(len(seq) - self.k+1, self.pad_length)):
                try:
                    sentence[i] = self.weights[seq[i:i+self.k]]
                except KeyError as ke:
                    if 'not in vocabulary' not in str(ke):
                        raise
            return sentence

        return array([transform_one(seq) for seq in X])

    def get_params(self):
        return dict(k=self.k, size=self.size, window=self.window,
                    pad_length=self.pad_length, workers=self.workers,
                    weights=self.weights)


class KmerEncoder(BaseEstimator):
    def __init__(self, k=7, size=8192, pad_length=None, encoder=None):
        self.k = k
        self.size = size
        self.pad_length = pad_length
        self.encoder = FeatureHasher(n_features=self.size,
                                     alternate_sign=False)

    def fit(self, X):
        pass

    def transform(self, X):
        def transform_one(seq):
            length = min(len(seq) - self.k+1, self.pad_length)
            kmers = [{seq[i:i+self.k]: 1} for i in range(length)]
            if length < self.pad_length:
                kmers += [dict()]*(self.pad_length - length)
            return self.encoder.transform(kmers)
        return array([transform_one(seq).todense() for seq in X])

    def get_params(self):
        return dict(k=self.k, size=self.size, pad_length=self.pad_length,
                    encoder=self.encoder)


# I'm keeping the spec_from_encoder and encoder_from_spec funtions for the
# moment, in case we want to enable them as specification inputs to
# fit-classifier-keras. But they're on notice.
def spec_from_encoder(encoder):
    class EncoderEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'get_params'):
                encoded = {}
                params = obj.get_params()
                subobjs = []
                for key, value in params.items():
                    if hasattr(value, 'get_params'):
                        subobjs.append(key + '__')

                for key, value in params.items():
                    for so in subobjs:
                        if key.startswith(so):
                            break
                    else:
                        if hasattr(value, 'get_params'):
                            encoded[key] = self.default(value)
                        try:
                            json.dumps(value, cls=EncoderEncoder)
                            encoded[key] = value
                        except TypeError:
                            pass

                module = obj.__module__
                type = module + '.' + obj.__class__.__name__
                encoded['__type__'] = type.split('.', 1)[1]
                return encoded
            return json.JSONEncoder.default(self, obj)
    return json.loads(json.dumps(encoder, cls=EncoderEncoder))


def encoder_from_spec(spec):
    def object_hook(obj):
        if '__type__' in obj:
            klass = _load_class(obj['__type__'])
            return klass(**{k: v for k, v in obj.items() if k != '__type__'})
        return obj
    return json.loads(json.dumps(spec), object_hook=object_hook)


class TaxonomicGenerator(KerasSequence):
    def __init__(self, X, y, X_encoder, y_encoder, batch_size):
        self.X = X
        self.y = y
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder
        self.batch_size = batch_size

    def __len__(self):
        return int(ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X_encoder.transform(batch_X)

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_encoder.transform(batch_y)

        return batch_X, batch_y


class XGenerator(KerasSequence):
    def __init__(self, X, X_encoder, batch_size):
        self.X = X
        self.X_encoder = X_encoder
        self.batch_size = batch_size

    def __len__(self):
        return int(ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X_encoder.transform(batch_X)

        return batch_X


def tensorflow_gpu_kludge():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


tensorflow_gpu_kludge()


def _unpack_class_weights(class_weight, encoder):
    class_weights = []
    for weights in class_weight.iter_data():
        class_weights.append(weights)
    if len(class_weights) > 1:
        raise ValueError(
                'Keras classifier does not support multilabel classification')
    weights = class_weights[0]
    encoded = encoder.transform([[c] for c in class_weight.ids('observation')])
    return encoded.T.dot(weights)


def fit_classifier_keras(reference_reads: DNAIterator,
                         reference_taxonomy: pd.Series,
                         classifier_specification: dict,
                         class_weight: biom.Table = None,
                         sequence_encoder: str = 'Seq2VecEncoder',
                         read_length: int = 300,
                         k: int = 7,
                         vec_length: int = 300,
                         window: int = 5,
                         n_jobs: int = 1,
                         loss: str = 'categorical_crossentropy',
                         optimizer: str = 'adam',
                         batch_size: int = 256,
                         epochs: int = 50) -> Klassifier:
    X, y = zip(*[(str(s), [reference_taxonomy[s.metadata['id']]])
                 for s in reference_reads
                 if s.metadata['id'] in reference_taxonomy])

    if sequence_encoder == 'DNAEncoder':
        x_encoder = DNAEncoder(read_length)
    elif sequence_encoder == 'Seq2VecEncoder':
        x_encoder = Seq2VecEncoder(k, vec_length, window, read_length, n_jobs)
    elif sequence_encoder == 'KmerEncoder':
        x_encoder = KmerEncoder(k, vec_length, read_length)

    y_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    x_encoder.fit(X)
    y_encoder.fit(y)

    if class_weight is not None:
        class_weight = _unpack_class_weights(class_weight, y_encoder)

    generator = TaxonomicGenerator(X, y, x_encoder, y_encoder, batch_size)
    # EEE So now we need to assert somewhere that keras_version is >= 2.3.0
    model = model_from_json(json.dumps(classifier_specification))
    classifier_specification = json.loads(model.to_json())  # to standardise
    classifier_specification['config']['layers'][-1]['config']['units'] = \
        y_encoder.transform([y[0]]).shape[1]
    model = model_from_json(json.dumps(classifier_specification))
    model.compile(loss=loss, optimizer=optimizer)
    model.fit_generator(generator, epochs=epochs, class_weight=class_weight)

    return Klassifier(x_encoder, y_encoder, model)


plugin.methods.register_function(
    function=fit_classifier_keras,
    inputs={'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy],
            'classifier_specification': ClassifierSpecification,
            'class_weight': FeatureTable[RelativeFrequency]},
    parameters={'sequence_encoder': Str % Choices(
        ['Seq2VecEncoder', 'DNAEncoder', 'KmerEncoder']),
                'read_length': Int,
                'k': Int,
                'vec_length': Int,
                'window': Int,
                'n_jobs': Int,
                'loss': Str,
                'optimizer': Str,
                'batch_size': Int,
                'epochs': Int},
    outputs=[('classifier', KerasClassifier)],
    name='Fit a Keras-based taxonomic classifier',
    description='Create a Keras classifier for reads'
)  # EEEE input_descriptions, parameter_descriptions, and citations


def fortify(y, encoder, confidence):
    downcoders = [encoder]
    downsamplers = []
    for i in encoder.categories_[0][0].split(';')[:-1]:
        taxa = [[t] for t in downcoders[-1].categories_[0]]
        taxa = [[';'.join(t.split(';')[:-1])] for t, in taxa]
        downcoders.append(OneHotEncoder())
        downcoders[-1].fit(taxa)
        downsamplers.append(downcoders[-1].transform(taxa).T)
    downsamplers.append(None)

    taxonomies = []
    confidences = []
    for output in y:
        for encoder, downsampler in zip(downcoders, downsamplers):
            if output.max() >= confidence:
                [[taxon]] = encoder.inverse_transform([output])
                break
            if downsampler is not None:
                output = downsampler.dot(output)
        else:
            taxon = _get_default_unassignable_label()
        taxonomies.append(taxon)
        confidences.append(output.max())

    return taxonomies, confidences


def classify_keras(reads: DNAIterator, classifier: Klassifier,
                   confidence: float = 0.7, batch_size: int = 256
                   ) -> pd.DataFrame:
    x_encoder, y_encoder, model = classifier
    X, seq_ids = zip(*[(str(s), s.metadata['id']) for s in reads])
    generator = XGenerator(X, x_encoder, batch_size)
    y = model.predict_generator(generator)
    if confidence == 'disable':
        taxonomy = [t for [t] in y_encoder.inverse_transform(y)]
        confidence = [-1]*len(y)
    else:
        taxonomy, confidence = fortify(y, y_encoder, confidence)

    result = pd.DataFrame(dict(Taxon=taxonomy, Confidence=confidence),
                          index=seq_ids, columns=['Taxon', 'Confidence'])
    result.index.name = 'Feature ID'
    return result


plugin.methods.register_function(
    function=classify_keras,
    inputs={'reads': FeatureData[Sequence],
            'classifier': KerasClassifier},
    parameters={'confidence': Float % Range(
        0, 1, inclusive_start=True, inclusive_end=True) | Str % Choices(
        ['disable']),
                'batch_size': Int},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Pre-fitted Keras-based taxonomy classifier',
    description='Classify reads by txon using a fitted classifier.'
)  # EEEE input_descriptions, parameter_descriptions, and citations
