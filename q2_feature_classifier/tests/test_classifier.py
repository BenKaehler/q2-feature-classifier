# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import os

from qiime2.sdk import Artifact
from q2_types.feature_data import DNAIterator
from qiime2.plugins import feature_classifier
import pandas as pd
import skbio
import biom

from q2_feature_classifier._skl import _specific_fitters
from q2_feature_classifier.classifier import spec_from_pipeline, \
    pipeline_from_spec, populate_class_weight, _autotune_reads_per_batch

from . import FeatureClassifierTestPluginBase


class ClassifierTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))

        self.seq_path = self.get_data_path('se-dna-sequences.fasta')
        reads = Artifact.import_data('FeatureData[Sequence]', self.seq_path)
        fitter_name = _specific_fitters[0][0]
        fitter = getattr(feature_classifier.methods,
                         'fit_classifier_' + fitter_name)
        self.classifier = fitter(reads, self.taxonomy).classifier

    def test_fit_classifier(self):
        # fit_classifier should generate a working taxonomic_classifier
        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

        classify = feature_classifier.methods.classify_sklearn
        result = classify(reads, self.classifier)

        ref = self.taxonomy.view(pd.Series).to_dict()
        classified = result.classification.view(pd.Series).to_dict()

        right = 0.
        for taxon in classified:
            right += ref[taxon].startswith(classified[taxon])
        self.assertGreater(right/len(classified), 0.95)

    def test_populate_class_weight(self):
        # should populate the class weight of a pipeline
        weights = Artifact.import_data(
            'FeatureTable[RelativeFrequency]',
            self.get_data_path('class_weight.biom'))
        table = weights.view(biom.Table)

        svc_spec = [['feat_ext',
                     {'__type__': 'feature_extraction.text.HashingVectorizer',
                      'analyzer': 'char_wb',
                      'n_features': 8192,
                      'ngram_range': [8, 8],
                      'alternate_sign': False}],
                    ['classify',
                     {'__type__': 'naive_bayes.GaussianNB'}]]
        pipeline1 = pipeline_from_spec(svc_spec)
        populate_class_weight(pipeline1, table)

        classes = table.ids('observation')
        class_weights = []
        for wts in table.iter_data():
            class_weights.append(zip(classes, wts))
        svc_spec[1][1]['priors'] = list(zip(*sorted(class_weights[0])))[1]
        pipeline2 = pipeline_from_spec(svc_spec)

        for a, b in zip(pipeline1.get_params()['classify__priors'],
                        pipeline2.get_params()['classify__priors']):
            self.assertAlmostEqual(a, b)

    def test_class_weight(self):
        # we should be able to input class_weight to fit_classifier
        weights = Artifact.import_data(
            'FeatureTable[RelativeFrequency]',
            self.get_data_path('class_weight.biom'))
        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

        fitter = feature_classifier.methods.fit_classifier_naive_bayes
        classifier1 = fitter(reads, self.taxonomy, class_weight=weights)
        classifier1 = classifier1.classifier

        class_weight = weights.view(biom.Table)
        classes = class_weight.ids('observation')
        class_weights = []
        for wts in class_weight.iter_data():
            class_weights.append(zip(classes, wts))
        priors = json.dumps(list(zip(*sorted(class_weights[0])))[1])
        classifier2 = fitter(reads, self.taxonomy,
                             classify__class_prior=priors).classifier

        classify = feature_classifier.methods.classify_sklearn
        result1 = classify(reads, classifier1)
        result1 = result1.classification.view(pd.Series).to_dict()
        result2 = classify(reads, classifier2)
        result2 = result2.classification.view(pd.Series).to_dict()
        self.assertEqual(result1, result2)

        svc_spec = [['feat_ext',
                     {'__type__': 'feature_extraction.text.HashingVectorizer',
                      'analyzer': 'char_wb',
                      'n_features': 8192,
                      'ngram_range': [8, 8],
                      'alternate_sign': False}],
                    ['classify',
                     {'__type__': 'linear_model.LogisticRegression'}]]
        classifier_spec = json.dumps(svc_spec)
        gen_fitter = feature_classifier.methods.fit_classifier_sklearn
        classifier1 = gen_fitter(reads, self.taxonomy, classifier_spec,
                                 class_weight=weights).classifier

        svc_spec[1][1]['class_weight'] = dict(class_weights[0])
        classifier_spec = json.dumps(svc_spec)
        gen_fitter = feature_classifier.methods.fit_classifier_sklearn
        classifier2 = gen_fitter(reads, self.taxonomy, classifier_spec
                                 ).classifier

        result1 = classify(reads, classifier1)
        result1 = result1.classification.view(pd.Series).to_dict()
        result2 = classify(reads, classifier2)
        result2 = result2.classification.view(pd.Series).to_dict()
        self.assertEqual(set(result1.keys()), set(result2.keys()))
        for k in result1:
            self.assertEqual(result1[k], result2[k])

    def test_fit_specific_classifiers(self):
        # specific and general classifiers should produce the same results
        gen_fitter = feature_classifier.methods.fit_classifier_sklearn
        classify = feature_classifier.methods.classify_sklearn
        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

        for name, spec in _specific_fitters:
            classifier_spec = json.dumps(spec)
            result = gen_fitter(reads, self.taxonomy, classifier_spec)
            result = classify(reads, result.classifier)
            gc = result.classification.view(pd.Series).to_dict()
            spec_fitter = getattr(feature_classifier.methods,
                                  'fit_classifier_' + name)
            result = spec_fitter(reads, self.taxonomy)
            result = classify(reads, result.classifier)
            sc = result.classification.view(pd.Series).to_dict()
            for taxon in gc:
                self.assertEqual(gc[taxon], sc[taxon])

    def test_pipeline_serialisation(self):
        # pipeline inflation and deflation should be inverse operations
        for name, spec in _specific_fitters:
            pipeline = pipeline_from_spec(spec)
            spec_one = spec_from_pipeline(pipeline)
            pipeline = pipeline_from_spec(spec_one)
            spec_two = spec_from_pipeline(pipeline)
            self.assertEqual(spec_one, spec_two)

    def test_classify(self):
        # test read direction detection and parallel classification
        classify = feature_classifier.methods.classify_sklearn
        seq_path = self.get_data_path('se-dna-sequences.fasta')
        reads = Artifact.import_data('FeatureData[Sequence]', seq_path)
        raw_reads = skbio.io.read(
            seq_path, format='fasta', constructor=skbio.DNA)
        rev_path = os.path.join(self.temp_dir.name, 'rev-dna-sequences.fasta')
        skbio.io.write((s.reverse_complement() for s in raw_reads),
                       'fasta', rev_path)
        rev_reads = Artifact.import_data('FeatureData[Sequence]', rev_path)

        result = classify(reads, self.classifier)
        fc = result.classification.view(pd.Series).to_dict()
        result = classify(rev_reads, self.classifier)
        rc = result.classification.view(pd.Series).to_dict()

        for taxon in fc:
            self.assertEqual(fc[taxon], rc[taxon])

        result = classify(reads, self.classifier, read_orientation='same')
        fc = result.classification.view(pd.Series).to_dict()
        result = classify(rev_reads, self.classifier,
                          read_orientation='reverse-complement')
        rc = result.classification.view(pd.Series).to_dict()

        for taxon in fc:
            self.assertEqual(fc[taxon], rc[taxon])

        result = classify(reads, self.classifier, reads_per_batch=100,
                          n_jobs=2)
        cc = result.classification.view(pd.Series).to_dict()

        for taxon in fc:
            self.assertEqual(fc[taxon], cc[taxon])

    def test_unassigned_taxa(self):
        # classifications that don't meet the threshold should be "Unassigned"
        classify = feature_classifier.methods.classify_sklearn
        seq_path = self.get_data_path('se-dna-sequences.fasta')
        reads = Artifact.import_data('FeatureData[Sequence]', seq_path)
        result = classify(reads, self.classifier, confidence=1.)

        ref = self.taxonomy.view(pd.Series).to_dict()
        classified = result.classification.view(pd.Series).to_dict()

        assert 'Unassigned' in classified.values()
        for seq in reads.view(DNAIterator):
            id_ = seq.metadata['id']
            assert ref[id_].startswith(classified[id_]) or \
                classified[id_] == 'Unassigned'

    def test_autotune_reads_per_batch(self):
        self.assertEqual(
            _autotune_reads_per_batch(self.seq_path, n_jobs=4), 276)

    def test_autotune_reads_per_batch_disable_if_single_job(self):
        self.assertEqual(
            _autotune_reads_per_batch(self.seq_path, n_jobs=1), 20000)

    def test_autotune_reads_per_batch_zero_jobs(self):
        with self.assertRaisesRegex(
                ValueError, "Value other than zero must be specified"):
            _autotune_reads_per_batch(self.seq_path, n_jobs=0)

    def test_autotune_reads_per_batch_ceil(self):
        self.assertEqual(
            _autotune_reads_per_batch(self.seq_path, n_jobs=5), 221)

    def test_autotune_reads_per_batch_more_jobs_than_reads(self):
        self.assertEqual(
            _autotune_reads_per_batch(self.seq_path, n_jobs=1105), 1)
