# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for BGCdetection.predictAndClassify batching logic."""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from sanntis.modules.BGCdetection import AnnotationFilesToEmerald

_N_VOCAB = 10
_CLAA = ["Alkaloid", "NRP", "Polyketide", "RiPP", "Saccharide", "Terpene", "Other"]

class _MockClassifier:
    """Scikit-learn classifier stub: returns 0.5 probability for all classes."""

    def predict_proba(self, X: list) -> np.ndarray:
        """Return uniform probabilities.

        :param X: Feature matrix (ignored).
        :type X: list
        :returns: Array of shape ``(n_samples, 2)`` filled with 0.5.
        :rtype: numpy.ndarray
        """
        return np.full((len(X), 2), 0.5)


def _make_mock_model() -> MagicMock:
    """Return a MagicMock that mimics a Keras model.

    ``predict`` returns an array of shape identical to the input filled with
    0.9, regardless of ``batch_size``.  This value exceeds the default
    detection threshold (0.855) so all CDSs are classified as BGC-positive,
    giving deterministic downstream results for all tests.

    :returns: Mock Keras model.
    :rtype: unittest.mock.MagicMock
    """
    model = MagicMock()

    def _predict(xva: np.ndarray, verbose: int = 0, batch_size: int = 32) -> np.ndarray:
        return np.full_like(xva, fill_value=0.9, dtype=np.float32)

    model.predict.side_effect = _predict
    return model


def _make_annotator() -> AnnotationFilesToEmerald:
    """Construct an :class:`AnnotationFilesToEmerald` with mocked model files.

    Patches ``builtins.open`` and ``pickle.load`` so no real model files are
    needed on disk.

    :returns: Freshly constructed annotator instance.
    :rtype: AnnotationFilesToEmerald
    """
    vocab = {f"IPR{i:03d}": i for i in range(_N_VOCAB)}
    type_model = {cla: _MockClassifier() for cla in _CLAA}
    # mbdoms entry: (mibig_id, bgc_class, frozenset_of_vocab_indices)
    mbdoms = [("BGC000001", "NRP", frozenset({0, 1, 2}))]

    with patch("builtins.open", mock_open()):
        with patch(
            "sanntis.modules.BGCdetection.pickle.load",
            return_value=(vocab, type_model, mbdoms),
        ):
            return AnnotationFilesToEmerald()


def _populate(ann: AnnotationFilesToEmerald, n_contigs: int = 3, cdss_per_contig: int = 5) -> None:
    """Fill *ann* with synthetic contig and domain annotation data.

    Each CDS is assigned one domain cycling through the vocabulary.

    :param ann: Annotator instance to populate.
    :type ann: AnnotationFilesToEmerald
    :param n_contigs: Number of contigs to create.
    :type n_contigs: int
    :param cdss_per_contig: Number of CDS features per contig.
    :type cdss_per_contig: int
    """
    ann.contigsDct = {}
    ann.entriesDct = {}
    for ci in range(n_contigs):
        contig_id = f"contig{ci}"
        cdss = []
        for pi in range(cdss_per_contig):
            pid = f"{contig_id}_{pi + 1}"
            start = pi * 300 + 1
            end = start + 299
            cdss.append((pid, (start, end)))
            ann.entriesDct[pid] = [f"IPR{(pi % _N_VOCAB):03d}"]
        ann.contigsDct[contig_id] = cdss


def _run_predict(
    ann: AnnotationFilesToEmerald,
    mock_model: MagicMock,
    chunk_size: int = None,
    batch_size: int = 32,
) -> None:
    """Call ``predictAndClassify`` with the given mock model.

    :param ann: Annotator instance.
    :type ann: AnnotationFilesToEmerald
    :param mock_model: Mock Keras model to inject.
    :type mock_model: unittest.mock.MagicMock
    :param chunk_size: Forwarded to ``contig_chunk_size``.
    :type chunk_size: int or None
    :param batch_size: Forwarded to ``batch_size``.
    :type batch_size: int
    """
    with patch(
        "sanntis.modules.BGCdetection.tf.keras.models.load_model",
        return_value=mock_model,
    ):
        ann.predictAndClassify(
            score=None,
            g=1,
            contig_chunk_size=chunk_size,
            batch_size=batch_size,
        )

@pytest.fixture
def ann() -> AnnotationFilesToEmerald:
    """Annotator instance with mocked model files and no contig data."""
    return _make_annotator()


def test_empty_input_does_not_raise(ann: AnnotationFilesToEmerald) -> None:
    """predictAndClassify on an annotator with no contigs should not raise.

    :param ann: Empty annotator fixture.
    :type ann: AnnotationFilesToEmerald
    """
    mock_model = _make_mock_model()
    _run_predict(ann, mock_model)

    assert ann.annResults == {}
    assert ann.looseClst == {}
    assert ann.borderClst == {}
    assert ann.typesClst == {}


def test_all_contigs_present_in_output(ann: AnnotationFilesToEmerald) -> None:
    """Every contig in contigsDct should appear in all output dicts.

    :param ann: Empty annotator fixture.
    :type ann: AnnotationFilesToEmerald
    """
    _populate(ann, n_contigs=3)
    _run_predict(ann, _make_mock_model())

    expected_contigs = {f"contig{i}" for i in range(3)}
    assert set(ann.annResults.keys()) == expected_contigs
    assert set(ann.looseClst.keys()) == expected_contigs
    assert set(ann.borderClst.keys()) == expected_contigs
    assert set(ann.typesClst.keys()) == expected_contigs


def test_model_predict_called_once_without_chunk_size(ann: AnnotationFilesToEmerald) -> None:
    """With no chunk size, all contigs are batched into a single model.predict call.

    :param ann: Empty annotator fixture.
    :type ann: AnnotationFilesToEmerald
    """
    _populate(ann, n_contigs=4)
    mock_model = _make_mock_model()
    _run_predict(ann, mock_model, chunk_size=None)

    assert mock_model.predict.call_count == 1


@pytest.mark.parametrize(
    "n_contigs,chunk_size,expected_calls",
    [
        (6, 2, 3),
        (6, 3, 2),
        (5, 2, 3),  # last chunk has 1 contig
        (1, 2, 1),  # fewer contigs than chunk size
    ],
)
def test_model_predict_call_count_with_chunk_size(
    n_contigs: int,
    chunk_size: int,
    expected_calls: int,
) -> None:
    """model.predict should be called once per chunk (ceiling division).

    :param n_contigs: Number of synthetic contigs.
    :type n_contigs: int
    :param chunk_size: Value for ``contig_chunk_size``.
    :type chunk_size: int
    :param expected_calls: Expected number of ``model.predict`` invocations.
    :type expected_calls: int
    """
    ann = _make_annotator()
    _populate(ann, n_contigs=n_contigs)
    mock_model = _make_mock_model()
    _run_predict(ann, mock_model, chunk_size=chunk_size)

    assert mock_model.predict.call_count == expected_calls


def test_batch_size_forwarded_to_model(ann: AnnotationFilesToEmerald) -> None:
    """The batch_size argument should be forwarded to every model.predict call.

    :param ann: Empty annotator fixture.
    :type ann: AnnotationFilesToEmerald
    """
    _populate(ann, n_contigs=2)
    mock_model = _make_mock_model()
    _run_predict(ann, mock_model, batch_size=16)

    for call in mock_model.predict.call_args_list:
        assert call.kwargs.get("batch_size") == 16


@pytest.mark.parametrize("chunk_size", [1, 2, None])
def test_results_identical_across_chunk_sizes(chunk_size: int) -> None:
    """annResults must be identical regardless of contig_chunk_size.

    The reference is produced with chunk_size=1 (one contig at a time).  All
    other chunk sizes must produce bit-for-bit identical annResults arrays,
    confirming that batching does not alter the per-contig predictions.

    :param chunk_size: chunk_size variant under test.
    :type chunk_size: int or None
    """
    n_contigs = 4
    mock_model = _make_mock_model()

    # --- reference: chunk_size=1 ---
    ref = _make_annotator()
    _populate(ref, n_contigs=n_contigs)
    _run_predict(ref, mock_model, chunk_size=1)
    ref_results = {k: v.copy() for k, v in ref.annResults.items()}

    # --- variant ---
    ann = _make_annotator()
    _populate(ann, n_contigs=n_contigs)
    _run_predict(ann, mock_model, chunk_size=chunk_size)

    assert set(ann.annResults.keys()) == set(ref_results.keys())
    for contig, ref_arr in ref_results.items():
        np.testing.assert_array_equal(
            ann.annResults[contig],
            ref_arr,
            err_msg=f"annResults differ for {contig} with chunk_size={chunk_size}",
        )
