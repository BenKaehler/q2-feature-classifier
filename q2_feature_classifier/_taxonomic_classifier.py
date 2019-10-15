# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import tarfile
import os

import sklearn
import joblib
from sklearn.pipeline import Pipeline
import qiime2.plugin
import qiime2.plugin.model as model

from .plugin_setup import plugin


# Semantic Types
TaxonomicClassifier = qiime2.plugin.SemanticType('TaxonomicClassifier')


# Formats
class PickleFormat(model.BinaryFileFormat):
    def sniff(self):
        return tarfile.is_tarfile(str(self))


# https://github.com/qiime2/q2-types/issues/49
class JSONFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            try:
                json.load(fh)
                return True
            except json.JSONDecodeError:
                pass
        return False


class TaxonomicClassifierDirFmt(model.DirectoryFormat):
    preprocess_params = model.File('preprocess_params.json', format=JSONFormat)
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


class TaxonomicClassiferTemporaryPickleDirFmt(model.DirectoryFormat):
    version_info = model.File('sklearn_version.json', format=JSONFormat)
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


# Transformers
@plugin.register_transformer
def _1(dirfmt: TaxonomicClassiferTemporaryPickleDirFmt) -> Pipeline:
    sklearn_version = dirfmt.version_info.view(object)['sklearn-version']
    if sklearn_version != sklearn.__version__:
        raise ValueError('The scikit-learn version (%s) used to generate this'
                         ' artifact does not match the current version'
                         ' of scikit-learn installed (%s). Please retrain your'
                         ' classifier for your current deployment to prevent'
                         ' data-corruption errors.'
                         % (sklearn_version, sklearn.__version__))

    sklearn_pipeline = dirfmt.sklearn_pipeline.view(PickleFormat)

    with tarfile.open(str(sklearn_pipeline)) as tar:
        tmpdir = model.DirectoryFormat()
        dirname = str(tmpdir)
        tar.extractall(dirname)
        pipeline = joblib.load(os.path.join(dirname, 'sklearn_pipeline.pkl'))
        for fn in tar.getnames():
            os.unlink(os.path.join(dirname, fn))

    return pipeline


@plugin.register_transformer
def _2(data: Pipeline) -> TaxonomicClassiferTemporaryPickleDirFmt:
    sklearn_pipeline = PickleFormat()
    with tarfile.open(str(sklearn_pipeline), 'w') as tar:
        tmpdir = model.DirectoryFormat()
        pf = os.path.join(str(tmpdir), 'sklearn_pipeline.pkl')
        for fn in joblib.dump(data, pf):
            tar.add(fn, os.path.basename(fn))
            os.unlink(fn)

    dirfmt = TaxonomicClassiferTemporaryPickleDirFmt()
    dirfmt.version_info.write_data(
        {'sklearn-version': sklearn.__version__}, object)
    dirfmt.sklearn_pipeline.write_data(sklearn_pipeline, PickleFormat)

    return dirfmt


@plugin.register_transformer
def _3(dirfmt: TaxonomicClassifierDirFmt) -> Pipeline:
    raise ValueError('The scikit-learn version could not be determined for'
                     ' this artifact, please retrain your classifier for your'
                     ' current deployment to prevent data-corruption errors.')


@plugin.register_transformer
def _4(fmt: JSONFormat) -> object:
    with fmt.open() as fh:
        return json.load(fh)


@plugin.register_transformer
def _5(data: object) -> JSONFormat:
    result = JSONFormat()
    with result.open() as fh:
        json.dump(data, fh)
    return result


# Registrations
plugin.register_semantic_types(TaxonomicClassifier)
plugin.register_formats(TaxonomicClassifierDirFmt,
                        TaxonomicClassiferTemporaryPickleDirFmt)
plugin.register_semantic_type_to_format(
    TaxonomicClassifier,
    artifact_format=TaxonomicClassiferTemporaryPickleDirFmt)
