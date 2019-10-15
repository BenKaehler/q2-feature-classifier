# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2.plugin
import qiime2.plugin.model as model
import h5py
from keras.model import load_model

from ._taxonomic_classifier import PickleFormat, JSONFormat
from .plugin_setup import plugin


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
    y_encoder = model.File('y_encoder.json', format=JSONFormat)
    x_encoder = model.File('x_encoder.json', format=JSONFormat)
    keras_model = model.File('keras_model.hdf5', format=KerasModelFormat)


# Transformers
@plugin.register_transformer
def _1(dirfmt: KerasClassifierDirectoryFomat) -> tuple:
    x_encoder = dirfmt.x_encoder.view(object)
    y_encoder = dirfmt.y_encoder.view(object)
    keras_model_format = dirfmt.keras_model.view(KerasModelFormat)
    keras_model = load_model(str(keras_model_format))
    return (x_encoder, y_encoder, keras_model)

@plugin.register_transformer
def _2(data: tuple) -> KerasClassifierDirectoryFormat:
    x_encoder, y_encoder, keras_model = data
    dirfmt = KerasClassifierDirectoryFormat()
    dirfmt.x_encoder.write_data(x_encoder, object)
    dirfmt.y_encoder.write_data(y_encoder, object)

    keras_model_format = KerasModelFormat()
    keras_model.save(str(keras_model_format))
    dirfmt.keras_model.write_data(keras_model_format, KerasModelFormat)

    return dirfmt

# Registrations
plugin.register_semantic_types(ClassifierSpecification)
plugin.register_formats(ClassifierSpecificationDirectoryFormat)
plugin.register_semantic_type_to_format(
    ClassifierSpecification,
    artifact_format=ClassifierSpecificationDirectoryFormat
)

