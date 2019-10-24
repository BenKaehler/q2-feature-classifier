# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import skbio
import os
from joblib import Parallel, delayed, effective_n_jobs

from qiime2.plugin import Int, Str, Float, Range, Choices
from q2_types.feature_data import (FeatureData, Sequence, DNAIterator,
                                   DNASequencesDirectoryFormat)
from q2_types.feature_data._format import DNAFASTAFormat
from q2_feature_classifier._skl import _chunks
from q2_feature_classifier.classifier import _autotune_reads_per_batch

from .plugin_setup import plugin


def _seq_to_regex(seq):
    """Build a regex out of a IUPAC consensus sequence"""
    result = []
    for base in str(seq):
        if base in skbio.DNA.degenerate_chars:
            result.append('[{0}]'.format(
                ''.join(skbio.DNA.degenerate_map[base])))
        else:
            result.append(base)

    return ''.join(result)


def _primers_to_regex(f_primer, r_primer):
    return '({0}.*{1})'.format(_seq_to_regex(f_primer),
                               _seq_to_regex(r_primer.reverse_complement()))


def _local_aln(primer, sequence):
    best_score = None
    for one_primer in primer.expand_degenerates():
        # `sequence` may contain degenerates. These will usually be N
        # characters, which SSW will score as zero. Although undocumented, SSW
        # will treat other degenerate characters as a mismatch. We acknowledge
        # that this approach is a heuristic to finding an optimal alignment and
        # may be revisited in the future if there's an aligner that explicitly
        # handles degenerates.
        this_aln = \
            skbio.alignment.local_pairwise_align_ssw(one_primer, sequence)
        score = this_aln[1]
        if best_score is None or score > best_score:
            best_score = score
            best_aln = this_aln
    return best_aln


def _semisemiglobal(primer, sequence, reverse=False):
    if reverse:
        primer = primer.reverse_complement()

    # locally align the primer
    (aln_prim, aln_seq), score, (prim_pos, seq_pos) = \
        _local_aln(primer, sequence)
    amplicon_pos = seq_pos[1]+len(primer)-prim_pos[1]

    # naively extend the alignment to be semi-global
    bits = [primer[:prim_pos[0]], aln_prim, primer[prim_pos[1]+1:]]
    aln_prim = ''.join(map(str, bits))
    bits = ['-'*(prim_pos[0]-seq_pos[0]),
            sequence[max(seq_pos[0]-prim_pos[0], 0):seq_pos[0]],
            aln_seq,
            sequence[seq_pos[1]+1:amplicon_pos],
            '-'*(amplicon_pos-len(sequence))]
    aln_seq = ''.join(map(str, bits))

    # count the matches
    matches = sum(s in skbio.DNA.degenerate_map.get(p, {p})
                  for p, s in zip(aln_prim, aln_seq))

    if reverse:
        amplicon_pos = max(seq_pos[0]-prim_pos[0], 0)

    return amplicon_pos, matches, len(aln_prim)


def _exact_match(seq, f_primer, r_primer):
    try:
        regex = _primers_to_regex(f_primer, r_primer)
        match = next(seq.find_with_regex(regex))
        beg, end = match.start + len(f_primer), match.stop - len(r_primer)
        return seq[beg:end]
    except StopIteration:
        return None


def _approx_match(seq, f_primer, r_primer, identity):
    beg, b_matches, b_length = _semisemiglobal(f_primer, seq)
    end, e_matches, e_length = _semisemiglobal(r_primer, seq, reverse=True)
    if (b_matches + e_matches) / (b_length + e_length) >= identity:
        return seq[beg:end]
    return None


def _gen_reads(sequence, f_primer, r_primer, trunc_len, trim_left, identity,
               min_length, max_length):
    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)
    amp = _exact_match(sequence, f_primer, r_primer)
    if not amp:
        amp = _exact_match(sequence.reverse_complement(), f_primer, r_primer)
    if not amp:
        amp = _approx_match(sequence, f_primer, r_primer, identity)
    if not amp:
        amp = _approx_match(
            sequence.reverse_complement(), f_primer, r_primer, identity)
    if not amp:
        return
    # we want to filter by max length before trimming
    if max_length > 0 and len(amp) > max_length:
        return
    if trunc_len > 0:
        amp = amp[:trunc_len]
    if trim_left > 0:
        amp = amp[trim_left:]
    if min_length > 0 and len(amp) < min_length:
        return
    if not amp:
        return
    return amp


def extract_reads(sequences: DNASequencesDirectoryFormat, f_primer: str,
                  r_primer: str, trunc_len: int = 0, trim_left: int = 0,
                  identity: float = 0.8, min_length: int = 50,
                  max_length: int = 0, n_jobs: int = 1,
                  batch_size: int = 'auto') -> DNAFASTAFormat:
    """Extract the read selected by a primer or primer pair. Only sequences
    which match the primers at greater than the specified identity are returned

    Parameters
    ----------
    sequences : DNASequencesDirectoryFormat
        An aligned list of skbio.sequence.DNA query sequences
    f_primer : skbio.sequence.DNA
        Forward primer sequence
    r_primer : skbio.sequence.DNA
        Reverse primer sequence
    trunc_len : int, optional
        Read is cut to trunc_len if trunc_len is positive. Applied before
        trim_left.
    trim_left : int, optional
        `trim_left` nucleotides are removed from the 5' end if trim_left is
        positive. Applied after trunc_len.
    identity : float, optional
        Minimum combined primer match identity threshold. Default: 0.8
    min_length: int, optional
        Minimum amplicon length. Shorter amplicons are discarded. Default: 50
    max_length: int, optional
        Maximum amplicon length. Longer amplicons are discarded.
    n_jobs: int, optional
        Number of seperate processes to break the task into.
    batch_size: int, optional
        Number of samples to be processed in one batch.
    Returns
    -------
    q2_types.DNAFASTAFormat
        containing the reads
    """
    if min_length > trunc_len - trim_left and trunc_len > 0:
        raise ValueError('The minimum length setting is greater than the '
                         'length of the truncated sequences. This will cause '
                         'all sequences to be removed from the dataset. To '
                         'proceed, set a min_length ≤ trunc_len - trim_left.')
    n_jobs = effective_n_jobs(n_jobs)
    if batch_size == 'auto':
        batch_size = _autotune_reads_per_batch(
            sequences.file.view(DNAFASTAFormat), n_jobs)
    sequences = sequences.file.view(DNAIterator)
    ff = DNAFASTAFormat()
    with open(str(ff), 'a') as fh:
        with Parallel(n_jobs) as parallel:
            for chunk in _chunks(sequences, batch_size):
                amplicons = parallel(delayed(_gen_reads)(sequence, f_primer,
                                                         r_primer, trunc_len,
                                                         trim_left, identity,
                                                         min_length,
                                                         max_length)
                                     for sequence in chunk)
                for amplicon in amplicons:
                    if amplicon is not None:
                        skbio.write(amplicon, format='fasta', into=fh)
    if os.stat(str(ff)).st_size == 0:
        raise RuntimeError("No matches found")
    return ff


plugin.methods.register_function(
    function=extract_reads,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={'trunc_len': Int,
                'trim_left': Int,
                'f_primer': Str,
                'r_primer': Str,
                'identity': Float,
                'min_length': Int % Range(0, None),
                'max_length': Int % Range(0, None),
                'n_jobs': Int % Range(1, None),
                'batch_size': Int % Range(1, None) | Str % Choices(['auto'])},
    outputs=[('reads', FeatureData[Sequence])],
    name='Extract reads from reference',
    description='Extract sequencing-like reads from a reference database.',
    parameter_descriptions={'f_primer': 'forward primer sequence',
                            'r_primer': 'reverse primer sequence',
                            'trunc_len': 'read is cut to trunc_len if '
                                         'trunc_len is positive. Applied '
                                         'before trim_left.',
                            'trim_left': 'trim_left nucleotides are removed '
                                         'from the 5\' end if trim_left is '
                                         'positive. Applied after trunc_len.',
                            'identity': 'minimum combined primer match '
                                        'identity threshold.',
                            'min_length': 'Minimum amplicon length. Shorter '
                                          'amplicons are discarded. Applied '
                                          'after trimming and truncation, so '
                                          'be aware that trimming may impact '
                                          'sequence retention. Set to zero '
                                          'to disable min length filtering.',
                            'max_length': 'Maximum amplicon length. Longer '
                                          'amplicons are discarded. Applied '
                                          'before trimming and truncation, '
                                          'so plan accordingly. Set to zero '
                                          '(default) to disable max length '
                                          'filtering.',
                            'n_jobs': 'Number of seperate processes to run.',
                            'batch_size': 'Number of sequences to process in '
                                          'a batch. The `auto` option is '
                                          'calculated from the number of '
                                          'sequences and number of jobs '
                                          'specified.'}
)
