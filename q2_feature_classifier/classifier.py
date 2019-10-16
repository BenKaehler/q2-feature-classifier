# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import importlib
import inspect
import warnings
from itertools import chain, islice
import subprocess

import pandas as pd
from qiime2.plugin import Int, Str, Float, Bool, Choices, Range
from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAIterator, DNAFASTAFormat)
from q2_types.feature_table import FeatureTable, RelativeFrequency
from sklearn.pipeline import Pipeline
import sklearn
from numpy import median, array, ceil
import biom
import skbio
import joblib

from ._skl import fit_pipeline, predict, _specific_fitters
from ._taxonomic_classifier import TaxonomicClassifier
from .plugin_setup import plugin, citations


def _load_class(classname):
    err_message = classname + ' is not a recognised class'
    if '.' not in classname:
        raise ValueError(err_message)
    module, klass = classname.rsplit('.', 1)
    if module == 'custom':
        module = importlib.import_module('.custom', 'q2_feature_classifier')
    elif module == '_keras':
        module = importlib.import_module('._keras', 'q2_feature_classifier')
    elif importlib.util.find_spec('.'+module, 'sklearn') is not None:
        module = importlib.import_module('.'+module, 'sklearn')
    else:
        raise ValueError(err_message)
    if not hasattr(module, klass):
        raise ValueError(err_message)
    klass = getattr(module, klass)
    if not issubclass(klass, sklearn.base.BaseEstimator):
        raise ValueError(err_message)
    return klass


def spec_from_pipeline(pipeline):
    class StepsEncoder(json.JSONEncoder):
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
                            json.dumps(value, cls=StepsEncoder)
                            encoded[key] = value
                        except TypeError:
                            pass

                module = obj.__module__
                type = module + '.' + obj.__class__.__name__
                encoded['__type__'] = type.split('.', 1)[1]
                return encoded
            return json.JSONEncoder.default(self, obj)
    steps = pipeline.get_params()['steps']
    return json.loads(json.dumps(steps, cls=StepsEncoder))


def pipeline_from_spec(spec):
    def as_steps(obj):
        if '__type__' in obj:
            klass = _load_class(obj['__type__'])
            return klass(**{k: v for k, v in obj.items() if k != '__type__'})
        return obj

    steps = json.loads(json.dumps(spec), object_hook=as_steps)
    return Pipeline(steps)


def warn_about_sklearn():
    warning = (
        'The TaxonomicClassifier artifact that results from this method was '
        'trained using scikit-learn version %s. It cannot be used with other '
        'versions of scikit-learn. (While the classifier may complete '
        'successfully, the results will be unreliable.)' % sklearn.__version__)
    warnings.warn(warning, UserWarning)


def populate_class_weight(pipeline, class_weight):
    classes = class_weight.ids('observation')
    class_weights = []
    for weights in class_weight.iter_data():
        class_weights.append(zip(classes, weights))
    step, classifier = pipeline.steps[-1]
    for param in classifier.get_params():
        if param == 'class_weight':
            class_weights = list(map(dict, class_weights))
            if len(class_weights) == 1:
                class_weights = class_weights[0]
            pipeline.set_params(**{'__'.join([step, param]): class_weights})
        elif param in ('priors', 'class_prior'):
            if len(class_weights) != 1:
                raise ValueError('naive_bayes classifiers do not support '
                                 'multilabel classification')
            priors = list(zip(*sorted(class_weights[0])))[1]
            pipeline.set_params(**{'__'.join([step, param]): priors})
    return pipeline


def fit_classifier_sklearn(reference_reads: DNAIterator,
                           reference_taxonomy: pd.Series,
                           classifier_specification: str,
                           class_weight: biom.Table = None) -> Pipeline:
    warn_about_sklearn()
    spec = json.loads(classifier_specification)
    pipeline = pipeline_from_spec(spec)
    if class_weight is not None:
        pipeline = populate_class_weight(pipeline, class_weight)
    pipeline = fit_pipeline(reference_reads, reference_taxonomy, pipeline)
    return pipeline


plugin.methods.register_function(
    function=fit_classifier_sklearn,
    inputs={'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy],
            'class_weight': FeatureTable[RelativeFrequency]},
    parameters={'classifier_specification': Str},
    outputs=[('classifier', TaxonomicClassifier)],
    name='Train an almost arbitrary scikit-learn classifier',
    description='Train a scikit-learn classifier to classify reads.',
    citations=[citations['pedregosa2011scikit']]
)


def _autodetect_orientation(reads, classifier, n=100,
                            read_orientation=None):
    reads = iter(reads)
    try:
        read = next(reads)
    except StopIteration:
        raise ValueError('empty reads input')
    if not hasattr(classifier, "predict_proba"):
        warnings.warn("this classifier does not support confidence values, "
                      "so read orientation autodetection is disabled",
                      UserWarning)
        return reads
    reads = chain([read], reads)
    if read_orientation == 'same':
        return reads
    if read_orientation == 'reverse-complement':
        return (r.reverse_complement() for r in reads)
    first_n_reads = list(islice(reads, n))
    result = list(zip(*predict(first_n_reads, classifier, confidence=0.)))
    _, _, same_confidence = result
    reversed_n_reads = [r.reverse_complement() for r in first_n_reads]
    result = list(zip(*predict(reversed_n_reads, classifier, confidence=0.)))
    _, _, reverse_confidence = result
    if median(array(same_confidence) - array(reverse_confidence)) > 0.:
        return chain(first_n_reads, reads)
    return chain(reversed_n_reads, (r.reverse_complement() for r in reads))


def _autotune_reads_per_batch(reads, n_jobs):
    # detect effective jobs. Will raise error if n_jobs == 0
    if n_jobs == 0:
        raise ValueError("Value other than zero must be specified as number "
                         "of jobs to run.")
    else:
        n_jobs = joblib.effective_n_jobs(n_jobs)

    # we really only want to calculate this if running in parallel
    if n_jobs != 1:
        seq_count = subprocess.run(
            ['grep', '-c', '^>', str(reads)], check=True,
            stdout=subprocess.PIPE)
        # set a max value to avoid blowing up memory
        return min(int(ceil(int(seq_count.stdout.decode('utf-8')) / n_jobs)),
                   20000)
    # otherwise reads_per_batch = 20000, which has a modest memory overhead
    else:
        return 20000


def classify_sklearn(reads: DNAFASTAFormat, classifier: Pipeline,
                     reads_per_batch: int = 0, n_jobs: int = 1,
                     pre_dispatch: str = '2*n_jobs', confidence: float = 0.7,
                     read_orientation: str = None
                     ) -> pd.DataFrame:
    # autotune reads per batch
    if reads_per_batch == 0:
        reads_per_batch = _autotune_reads_per_batch(reads, n_jobs)

    # transform reads to DNAIterator
    reads = DNAIterator(
        skbio.read(str(reads), format='fasta', constructor=skbio.DNA))

    reads = _autodetect_orientation(
        reads, classifier, read_orientation=read_orientation)
    predictions = predict(reads, classifier, chunk_size=reads_per_batch,
                          n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                          confidence=confidence)
    seq_ids, taxonomy, confidence = list(zip(*predictions))
    result = pd.DataFrame({'Taxon': taxonomy, 'Confidence': confidence},
                          index=seq_ids, columns=['Taxon', 'Confidence'])
    result.index.name = 'Feature ID'
    return result


_classify_parameters = {
    'reads_per_batch': Int % Range(0, None),
    'n_jobs': Int,
    'pre_dispatch': Str,
    'confidence': Float,
    'read_orientation': Str % Choices(['same', 'reverse-complement'])}


plugin.methods.register_function(
    function=classify_sklearn,
    inputs={'reads': FeatureData[Sequence],
            'classifier': TaxonomicClassifier},
    parameters=_classify_parameters,
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Pre-fitted sklearn-based taxonomy classifier',
    description='Classify reads by taxon using a fitted classifier.',
    input_descriptions={
        'reads': 'The feature data to be classified.',
        'classifier': 'The taxonomic classifier for classifying the reads.'
    },
    parameter_descriptions={
        'confidence': 'Confidence threshold for limiting '
                      'taxonomic depth. Provide -1 to disable '
                      'confidence calculation, or 0 to calculate '
                      'confidence but not apply it to limit the '
                      'taxonomic depth of the assignments.',
        'read_orientation': 'Direction of reads with '
                            'respect to reference sequences. same will cause '
                            'reads to be classified unchanged; reverse-'
                            'complement will cause reads to be reversed '
                            'and complemented prior to classification. '
                            'Default is to autodetect based on the '
                            'confidence estimates for the first 100 reads.',
        'reads_per_batch': 'Number of reads to process in each batch. If 0, '
                           'this parameter is autoscaled to '
                           'min( number of query sequences / n_jobs, 20000).',
        'n_jobs': 'The maximum number of concurrently worker processes. If -1 '
                  'all CPUs are used. If 1 is given, no parallel computing '
                  'code is used at all, which is useful for debugging. For '
                  'n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for '
                  'n_jobs = -2, all CPUs but one are used.',
        'pre_dispatch': '"all" or expression, as in "3*n_jobs". The number of '
                        'batches (of tasks) to be pre-dispatched.'
    },
    citations=[citations['pedregosa2011scikit']]
)


def _pipeline_signature(spec):
    type_map = {int: Int, float: Float, bool: Bool, str: Str}
    parameters = {}
    signature_params = []
    pipeline = pipeline_from_spec(spec)
    params = pipeline.get_params()
    for param, default in sorted(params.items()):
        # weed out pesky memory parameter from skl
        # https://github.com/qiime2/q2-feature-classifier/issues/101
        if param == 'memory':
            continue
        try:
            json.dumps(default)
        except TypeError:
            continue
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        if type(default) in type_map:
            annotation = type(default)
        else:
            annotation = str
            default = json.dumps(default)
        new_param = inspect.Parameter(param, kind, default=default,
                                      annotation=annotation)
        signature_params.append(new_param)
        parameters[param] = type_map.get(annotation, Str)
    return parameters, signature_params


def _register_fitter(name, spec):
    parameters, signature_params = _pipeline_signature(spec)

    def generic_fitter(reference_reads: DNAIterator,
                       reference_taxonomy: pd.Series,
                       class_weight: biom.Table = None, **kwargs) -> Pipeline:
        warn_about_sklearn()
        for param in kwargs:
            try:
                kwargs[param] = json.loads(kwargs[param])
            except (json.JSONDecodeError, TypeError):
                pass
        pipeline = pipeline_from_spec(spec)
        pipeline.set_params(**kwargs)
        if class_weight is not None:
            pipeline = populate_class_weight(pipeline, class_weight)
        pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                                pipeline)
        return pipeline

    generic_signature = inspect.signature(generic_fitter)
    new_params = list(generic_signature.parameters.values())[:-1]
    new_params.extend(signature_params)
    return_annotation = generic_signature.return_annotation
    new_signature = inspect.Signature(parameters=new_params,
                                      return_annotation=return_annotation)
    generic_fitter.__signature__ = new_signature
    generic_fitter.__name__ = 'fit_classifier_' + name
    plugin.methods.register_function(
        function=generic_fitter,
        inputs={'reference_reads': FeatureData[Sequence],
                'reference_taxonomy': FeatureData[Taxonomy],
                'class_weight': FeatureTable[RelativeFrequency]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the ' + name + ' classifier',
        description='Create a scikit-learn ' + name + ' classifier for reads',
        citations=[citations['pedregosa2011scikit']]
    )


for name, pipeline in _specific_fitters:
    _register_fitter(name, pipeline)
