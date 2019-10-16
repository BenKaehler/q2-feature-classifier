# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import tarfile
from collections import namedtuple
import os

import qiime2.plugin
import qiime2.plugin.model as model
import h5py
from keras.models import load_model
import joblib

from ._taxonomic_classifier import JSONFormat, PickleFormat
from .plugin_setup import plugin


Klassifier = namedtuple('Klassifier', ['x_encoder', 'y_encoder', 'model'])


# Semantic Types
ClassifierSpecification = qiime2.plugin.SemanticType('ClassifierSpecification')
KerasClassifier = qiime2.plugin.SemanticType('KerasClassifier')

# Formats
ClassifierSpecificationDirectoryFormat = model.SingleFileDirectoryFormat(
    'ClassifierSpecificationDirectoryFormat', 'classifier-specification.json',
    JSONFormat)


class KerasModelFormat(model.BinaryFileFormat):
    def sniff(self):
        return h5py.is_hdf5(str(self))


class KerasClassifierDirectoryFormat(model.DirectoryFormat):
    y_encoder = model.File('y_encoder.tar', format=PickleFormat)
    x_encoder = model.File('x_encoder.tar', format=PickleFormat)
    keras_model = model.File('keras_model.hdf5', format=KerasModelFormat)


# Transformers
def _enloader(encoder, pkl_file):
    encoder_pkl = encoder.view(PickleFormat)
    with tarfile.open(str(encoder_pkl)) as tar:
        tmpdir = model.DirectoryFormat()
        dirname = str(tmpdir)
        tar.extractall(dirname)
        encoder = joblib.load(os.path.join(dirname, pkl_file))
        for fn in tar.getnames():
            os.unlink(os.path.join(dirname, fn))
    return encoder


def _unloader(encoder, pkl_file):
    encoder_pkl = PickleFormat()
    with tarfile.open(str(encoder_pkl), 'w') as tar:
        tmpdir = model.DirectoryFormat()
        pf = os.path.join(str(tmpdir), pkl_file)
        for fn in joblib.dump(encoder, pf):
            tar.add(fn, os.path.basename(fn))
            os.unlink(fn)
    return encoder_pkl


@plugin.register_transformer
def _1(dirfmt: KerasClassifierDirectoryFormat) -> Klassifier:
    x_encoder = _enloader(dirfmt.x_encoder, 'x_encoder.pkl')
    y_encoder = _enloader(dirfmt.y_encoder, 'y_encoder.pkl')
    keras_model_format = dirfmt.keras_model.view(KerasModelFormat)
    keras_model = load_model(str(keras_model_format))
    return Klassifier(x_encoder, y_encoder, keras_model)


@plugin.register_transformer
def _2(data: Klassifier) -> KerasClassifierDirectoryFormat:

    dirfmt = KerasClassifierDirectoryFormat()
    x_encoder, y_encoder, keras_model = data
    x_encoder_pkl = _unloader(x_encoder, 'x_encoder.pkl')
    y_encoder_pkl = _unloader(y_encoder, 'y_encoder.pkl')
    dirfmt.x_encoder.write_data(x_encoder_pkl, PickleFormat)
    dirfmt.y_encoder.write_data(y_encoder_pkl, PickleFormat)

    keras_model_format = KerasModelFormat()
    keras_model.save(str(keras_model_format))
    dirfmt.keras_model.write_data(keras_model_format, KerasModelFormat)

    return dirfmt


# Registrations
plugin.register_semantic_types(ClassifierSpecification)
plugin.register_formats(ClassifierSpecificationDirectoryFormat,
                        KerasClassifierDirectoryFormat)
plugin.register_semantic_type_to_format(
    ClassifierSpecification,
    artifact_format=ClassifierSpecificationDirectoryFormat
)
plugin.register_semantic_type_to_format(
    KerasClassifier,
    artifact_format=KerasClassifierDirectoryFormat
)
