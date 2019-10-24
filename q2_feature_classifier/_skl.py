# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import islice, repeat
from copy import deepcopy

from joblib import Parallel, delayed

_specific_fitters = [
        ['naive_bayes',
         [['feat_ext',
           {'__type__': 'feature_extraction.text.HashingVectorizer',
            'analyzer': 'char_wb',
            'n_features': 8192,
            'ngram_range': [7, 7],
            'alternate_sign': False}],
          ['classify',
           {'__type__': 'custom.LowMemoryMultinomialNB',
            'alpha': 0.001,
            'fit_prior': False}]]]]


def fit_pipeline(reads, taxonomy, pipeline):
    seq_ids, X = _extract_reads(reads)
    data = [(taxonomy[s], x) for s, x in zip(seq_ids, X) if s in taxonomy]
    y, X = list(zip(*data))
    pipeline.fit(X, y)
    return pipeline


def _extract_reads(reads):
    return zip(*[(r.metadata['id'], r._string) for r in reads])


def predict(reads, pipeline, separator=';', chunk_size=262144, n_jobs=1,
            pre_dispatch='2*n_jobs', confidence='disable'):
    return (m for c in Parallel(n_jobs=n_jobs, batch_size=1,
                                pre_dispatch=pre_dispatch)
            (delayed(_predict_chunk)(pipeline, separator, confidence, chunk)
             for chunk in _chunks(reads, chunk_size)) for m in c)


def _predict_chunk(pipeline, separator, confidence, chunk):
    if confidence == 'disable':
        return _predict_chunk_without_conf(pipeline, chunk)
    else:
        return _predict_chunk_with_conf(pipeline, separator, confidence, chunk)


def _predict_chunk_without_conf(pipeline, chunk):
    seq_ids, X = _extract_reads(chunk)
    y = pipeline.predict(X)
    return zip(seq_ids, y, repeat(-1.))


def _predict_chunk_with_conf(pipeline, separator, confidence, chunk):
    seq_ids, X = _extract_reads(chunk)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError('this classifier does not support confidence values')
    prob_pos = pipeline.predict_proba(X)
    if prob_pos.shape != (len(X), len(pipeline.classes_)):
        raise ValueError('this classifier does not support confidence values')

    y = pipeline.classes_[prob_pos.argmax(axis=1)]

    results = []
    split_classes = [c.split(separator) for c in pipeline.classes_]
    for seq_id, taxon, class_probs in zip(seq_ids, y, prob_pos):
        taxon = taxon.split(separator)
        classes = zip(deepcopy(split_classes), class_probs)
        result = []
        for level in taxon:
            classes = [cls for cls in classes if cls[0].pop(0) == level]
            cum_prob = sum(c[1] for c in classes)
            if cum_prob < confidence:
                break
            result.append(level)
            result_confidence = cum_prob
        if result:
            result = separator.join(result)
            results.append((seq_id, result, result_confidence))
        else:
            results.append((seq_id, 'Unassigned', 1. - cum_prob))

    return results


def _chunks(reads, chunk_size):
    reads = iter(reads)
    while True:
        chunk = list(islice(reads, chunk_size))
        if len(chunk) == 0:
            break
        yield chunk
