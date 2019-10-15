# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2.plugin
import qiime2.plugin.model as model

from ._taxonomic_classifier import PickleFormat, JSONFormat
from .plugin_setup import plugin


# Semantic Types
ClassifierSpecification = qiime2.plugin.SemanticType('ClassifierSpecification')
KerasClassifier = qiime2.plugin.SemanticType('KerasClassifier')

class ClassifierSpecificationDirFmt(model.DirectoryFormat):
    classifier_specification = model.File(
        'classifier_specification.json', format=JSONFormat)

class KerasClassifierPickleDirFmt(model.DirectoryFormat):
    y_encoder = model.File('y_encoder.json', format=JSONFormat)
    x_encoder = model.File('x_encoder.json', format=JSONFormat)
    keras_pipeline = model.File('keras_model.tar', format=PickleFormat)


# Transformers
@plugin.regist_transformer
def _1(dirfmt: KerasClassifierPickleDirFmt) -> KerasPipeline:
    x_encoder_spec = dirfmt.x_encoder.view(object)
    y_encoder_spec = dirfmt.y_encoder.view(object)

