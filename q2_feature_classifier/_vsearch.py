# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import tempfile
import pandas as pd
import qiime2

from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAFASTAFormat)
from .plugin_setup import plugin, citations
from qiime2.plugin import Int, Str, Float, Choices, Range, Bool
from ._consensus_assignment import (_consensus_assignments, _run_command,
                                    _get_default_unassignable_label,
                                    _annotate_method)
from ._taxonomic_classifier import TaxonomicClassifier
from .classifier import _classify_parameters, _parameter_descriptions


def classify_consensus_vsearch(query: DNAFASTAFormat,
                               reference_reads: DNAFASTAFormat,
                               reference_taxonomy: pd.Series,
                               maxaccepts: int = 10,
                               perc_identity: float = 0.8,
                               query_cov: float = 0.8,
                               strand: str = 'both',
                               min_consensus: float = 0.51,
                               unassignable_label: str =
                               _get_default_unassignable_label(),
                               search_exact: bool = False,
                               top_hits_only: bool = False,
                               threads: str = 1) -> pd.DataFrame:
    seqs_fp = str(query)
    ref_fp = str(reference_reads)
    if maxaccepts == 'all':
        maxaccepts = 0
    cmd = ['vsearch', '--usearch_global', seqs_fp, '--id', str(perc_identity),
           '--query_cov', str(query_cov), '--strand', strand, '--maxaccepts',
           str(maxaccepts), '--maxrejects', '0', '--output_no_hits', '--db',
           ref_fp, '--threads', str(threads)]
    if search_exact:
        cmd[1] = '--search_exact'
    if top_hits_only:
        cmd.append('--top_hits_only')
    cmd.append('--blast6out')
    consensus = _consensus_assignments(
        cmd, reference_taxonomy, min_consensus=min_consensus,
        unassignable_label=unassignable_label)
    return consensus


def classify_hybrid_vsearch_sklearn(ctx,
                                    query,
                                    reference_reads,
                                    reference_taxonomy,
                                    classifier,
                                    maxaccepts=10,
                                    perc_identity=0.5,
                                    query_cov=0.8,
                                    strand='both',
                                    min_consensus=0.51,
                                    reads_per_batch=0,
                                    confidence=0.7,
                                    read_orientation='auto',
                                    threads=1,
                                    prefilter=True,
                                    sample_size=1000,
                                    randseed=0):
    exclude = ctx.get_action('quality_control', 'exclude_seqs')
    ccv = ctx.get_action('feature_classifier', 'classify_consensus_vsearch')
    cs = ctx.get_action('feature_classifier', 'classify_sklearn')
    filter_seqs = ctx.get_action('taxa', 'filter_seqs')
    merge = ctx.get_action('feature_table', 'merge_taxa')

    # randomly subsample reference sequences for rough positive filter
    if prefilter:
        ref = str(reference_reads.view(DNAFASTAFormat))
        with tempfile.NamedTemporaryFile() as output:
            cmd = ['vsearch', '--fastx_subsample', ref, '--sample_size',
                   str(sample_size), '--randseed', str(randseed),
                   '--fastaout', output.name]
            _run_command(cmd)
            sparse_reference = qiime2.Artifact.import_data(
                'FeatureData[Sequence]', output.name)

            # perform rough positive filter on query sequences
            query, misses, = exclude(
                query_sequences=query, reference_sequences=sparse_reference,
                method='vsearch', perc_identity=perc_identity,
                perc_query_aligned=query_cov, threads=threads)

    # find exact matches, perform LCA consensus classification
    taxa1, = ccv(query=query, reference_reads=reference_reads,
                 reference_taxonomy=reference_taxonomy, maxaccepts=maxaccepts,
                 strand=strand, min_consensus=min_consensus,
                 search_exact=True, threads=threads)

    # Annotate taxonomic assignments with classification method
    taxa1 = _annotate_method(taxa1, 'VSEARCH')

    # perform second pass classification on unassigned taxa
    # filter out unassigned seqs
    try:
        query, = filter_seqs(sequences=query, taxonomy=taxa1,
                             include=_get_default_unassignable_label())
    except ValueError:
        # get ValueError if all sequences are filtered out.
        # so if no sequences are unassigned, return exact match results
        return taxa1

    # classify with sklearn classifier
    taxa2, = cs(reads=query, classifier=classifier,
                reads_per_batch=reads_per_batch, n_jobs=threads,
                confidence=confidence, read_orientation=read_orientation)

    # Annotate taxonomic assignments with classification method
    taxa2 = _annotate_method(taxa2, 'sklearn')

    # merge into one big happy result
    taxa, = merge(data=[taxa2, taxa1])
    return taxa


output_descriptions = {
    'classification': 'The resulting taxonomy classifications.'}

parameters = {'maxaccepts': Int % Range(1, None) | Str % Choices(['all']),
              'perc_identity': Float % Range(0.0, 1.0, inclusive_end=True),
              'query_cov': Float % Range(0.0, 1.0, inclusive_end=True),
              'strand': Str % Choices(['both', 'plus']),
              'min_consensus': Float % Range(0.5, 1.0, inclusive_end=True,
                                             inclusive_start=False),
              'threads': Int % Range(1, None)}

inputs = {'query': FeatureData[Sequence],
          'reference_reads': FeatureData[Sequence],
          'reference_taxonomy': FeatureData[Taxonomy]}

input_descriptions = {'query': 'Sequences to classify taxonomically.',
                      'reference_reads': 'reference sequences.',
                      'reference_taxonomy': 'reference taxonomy labels.'}

parameter_descriptions = {
    'strand': 'Align against reference sequences in forward ("plus") '
              'or both directions ("both").',
    'maxaccepts': 'Maximum number of hits to keep for each query. Set to '
                  '"all" to keep all hits > perc_identity similarity.',
    'perc_identity': 'Reject match if percent identity to query is '
                     'lower.',
    'query_cov': 'Reject match if query alignment coverage per high-'
                 'scoring pair is lower.',
    'min_consensus': 'Minimum fraction of assignments must match top '
                     'hit to be accepted as consensus assignment.',
    'threads': 'Number of threads to use for job parallelization.'}

outputs = [('classification', FeatureData[Taxonomy])]

ignore_prefilter = ' This parameter is ignored if `prefilter` is disabled.'


plugin.methods.register_function(
    function=classify_consensus_vsearch,
    inputs=inputs,
    parameters={**parameters,
                'unassignable_label': Str,
                'search_exact': Bool,
                'top_hits_only': Bool},
    outputs=outputs,
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions,
        'search_exact': 'Search for exact full-length matches to the query '
                        'sequences. Only 100% exact matches are reported and '
                        'this command is much faster than the default. If '
                        'True, the perc_identity and query_cov settings are '
                        'ignored. Note: query and reference reads must be '
                        'trimmed to the exact same DNA locus (e.g., primer '
                        'site) because only exact matches will be reported.',
        'top_hits_only': 'Only the top hits between the query and reference '
                         'sequence sets are reported. For each query, the top '
                         'hit is the one presenting the highest percentage of '
                         'identity. Multiple equally scored top hits will be '
                         'used for consensus taxonomic assignment if '
                         'maxaccepts is greater than 1.',
    },
    output_descriptions=output_descriptions,
    name='VSEARCH-based consensus taxonomy classifier',
    description=('Assign taxonomy to query sequences using VSEARCH. Performs '
                 'VSEARCH global alignment between query and reference_reads, '
                 'then assigns consensus taxonomy to each query sequence from '
                 'among maxaccepts top hits, min_consensus of which share '
                 'that taxonomic assignment. Unlike classify-consensus-blast, '
                 'this method searches the entire reference database before '
                 'choosing the top N hits, not the first N hits.'),
    citations=[citations['rognes2016vsearch']]
)


plugin.pipelines.register_function(
    function=classify_hybrid_vsearch_sklearn,
    inputs={**inputs, 'classifier': TaxonomicClassifier},
    parameters={**parameters,
                'reads_per_batch': _classify_parameters['reads_per_batch'],
                'confidence': _classify_parameters['confidence'],
                'read_orientation': _classify_parameters['read_orientation'],
                'prefilter': Bool,
                'sample_size': Int % Range(1, None),
                'randseed': Int % Range(0, None)},
    outputs=outputs,
    input_descriptions={**input_descriptions,
                        'classifier': 'Pre-trained sklearn taxonomic '
                                      'classifier for classifying the reads.'},
    parameter_descriptions={
        **{k: parameter_descriptions[k] for k in [
            'strand', 'maxaccepts', 'min_consensus', 'threads']},
        'perc_identity': 'Percent sequence similarity to use for PREFILTER. ' +
                         parameter_descriptions['perc_identity'] + ' Set to a '
                         'lower value to perform a rough pre-filter.' +
                         ignore_prefilter,
        'query_cov': 'Query coverage threshold to use for PREFILTER. ' +
                     parameter_descriptions['query_cov'] + ' Set to a '
                     'lower value to perform a rough pre-filter.' +
                     ignore_prefilter,
        'confidence': _parameter_descriptions['confidence'],
        'read_orientation': 'Direction of reads with respect to reference '
                            'sequences in pre-trained sklearn classifier. '
                            'same will cause reads to be classified unchanged'
                            '; reverse-complement will cause reads to be '
                            'reversed and complemented prior to '
                            'classification. "auto" will autodetect '
                            'orientation based on the confidence estimates '
                            'for the first 100 reads.',
        'reads_per_batch': 'Number of reads to process in each batch for '
                           'sklearn classification. If "auto", this parameter '
                           'is autoscaled to min(number of query sequences / '
                           'threads, 20000).',
        'prefilter': 'Toggle positive filter of query sequences on or off.',
        'sample_size': 'Randomly extract the given number of sequences from '
                       'the reference database to use for prefiltering.' +
                       ignore_prefilter,
        'randseed': 'Use integer as a seed for the pseudo-random generator '
                    'used during prefiltering. A given seed always produces '
                    'the same output, which is useful for replicability. Set '
                    'to 0 to use a pseudo-random seed.' + ignore_prefilter,
    },
    output_descriptions=output_descriptions,
    name='ALPHA Hybrid classifier: VSEARCH exact match + sklearn classifier',
    description=('NOTE: THIS PIPELINE IS AN ALPHA RELEASE. Please report bugs '
                 'to https://forum.qiime2.org!\n'
                 'Assign taxonomy to query sequences using hybrid classifier. '
                 'First performs rough positive filter to remove artifact and '
                 'low-coverage sequences (use "prefilter" parameter to toggle '
                 'this step on or off). Second, performs VSEARCH exact match '
                 'between query and reference_reads to find exact matches, '
                 'followed by least common ancestor consensus taxonomy '
                 'assignment from among maxaccepts top hits, min_consensus of '
                 'which share that taxonomic assignment. Query sequences '
                 'without an exact match are then classified with a pre-'
                 'trained sklearn taxonomy classifier to predict the most '
                 'likely taxonomic lineage.'),
)
