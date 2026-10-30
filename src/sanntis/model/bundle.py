#!/usr/bin/env python3

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

"""Model bundle: every stage in one directory, loaded the same way from HF Hub or a local path.

```
<bundle>/
  config.json  vocab.json  signature_map.parquet
  detector-seed{0,1,2}.safetensors  postfilter.skops
  npc.skops  npc_knn.npz  npc_thresholds.json      (absent when the NPC head is disabled)
  npc_key_domains.json                             (optional: NPC pathway key-domain rules)
  manifest.json  README.md  report/
```
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from sanntis._version import __version__
from sanntis.config import DecodeConfig, DetectorConfig, FeaturesConfig, load_recipe
from sanntis.features.vocab import Vocab
from sanntis.model.detector import Detector
from sanntis.model.npc import NpcModel
from sanntis.model.postfilter import PostFilter
from sanntis.utils import sha256_file

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_REPO = "microbiome-informatics/sanntis"
DEFAULT_REVISION = "v1.0.0-beta.1"  # Hub tag of the default model = git tag of this release; bump with _version.py
SIGNATURE_MAP_COLUMNS = ["signature", "app", "entry"]


class BundleError(RuntimeError):
    pass


class IpsMismatchError(BundleError):
    pass


def default_model_id() -> str:
    """The Hub release matching the installed sanntis version."""
    return f"{DEFAULT_REPO}@{DEFAULT_REVISION}"


@dataclass
class BundleConfig:
    """``config.json``: everything inference needs besides the weights."""

    detector: DetectorConfig
    features: FeaturesConfig
    decode: DecodeConfig  # decode.threshold is the run's operating point t*
    postfilter_drop_threshold: float
    bgc_classes: list[str]
    npc_enabled: bool = True
    class_enabled: bool = True
    coverage_reference: dict = field(default_factory=dict)  # training QC values
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict:
        return {"schema_version": self.schema_version, "detector": self.detector.model_dump(),
                "features": self.features.model_dump(), "decode": self.decode.model_dump(),
                "postfilter": {"drop_threshold": self.postfilter_drop_threshold, "classes": self.bgc_classes},
                "heads": {"npc": self.npc_enabled, "class": self.class_enabled},
                "coverage_reference": self.coverage_reference}

    @classmethod
    def from_json(cls, d: dict) -> BundleConfig:
        return cls(detector=DetectorConfig(**d["detector"]), features=FeaturesConfig(**d["features"]),
                   decode=DecodeConfig(**d["decode"]), postfilter_drop_threshold=d["postfilter"]["drop_threshold"],
                   bgc_classes=list(d["postfilter"]["classes"]), npc_enabled=d["heads"]["npc"],
                   class_enabled=d["heads"]["class"],
                   coverage_reference=d.get("coverage_reference", {}), schema_version=d["schema_version"])


@dataclass
class SanntisModel:
    config: BundleConfig
    vocab: Vocab
    signature_map: pd.DataFrame
    detector_states: list[dict[str, torch.Tensor]]
    postfilter: PostFilter
    npc: NpcModel | None
    manifest: dict = field(default_factory=dict)
    report_dir: Path | None = None
    source: str = ""
    _detectors: list[Detector] | None = field(default=None, repr=False)

    # ── loading ──────────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, model_id: str | os.PathLike | None = None, revision: str | None = None, *,
                        allow_ips_mismatch: bool = False, cache_dir: str | os.PathLike | None = None,
                        verify: bool = True) -> SanntisModel:
        """Load a bundle from a local directory or a Hub repo (``repo`` or ``repo@revision``)."""
        model_id = str(model_id or default_model_id())
        path = Path(model_id).expanduser()
        if path.is_dir():
            local, source = path, str(path.resolve())
        else:
            repo, _, rev = model_id.partition("@")
            rev = revision or rev or None
            if rev is None:
                raise BundleError(f"{model_id}: Hub models must be pinned to a revision (repo@rev)")
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(repo_id=repo, revision=rev, cache_dir=cache_dir))
            source = f"{repo}@{rev}"
        return cls.load(local, allow_ips_mismatch=allow_ips_mismatch, verify=verify, source=source)

    @classmethod
    def load(cls, bundle: Path, *, allow_ips_mismatch: bool = False, verify: bool = True,
             source: str = "") -> SanntisModel:
        bundle = Path(bundle)
        mf = bundle / "manifest.json"
        if not mf.is_file():
            raise BundleError(f"{bundle}: no manifest.json (not a sanntis bundle)")
        manifest = json.loads(mf.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(f"{bundle}: bundle schema {manifest.get('schema_version')} is not supported "
                              f"by sanntis {__version__} (schema {SCHEMA_VERSION})")
        if verify:
            verify_files(bundle, manifest)
        check_ips(manifest, allow_mismatch=allow_ips_mismatch)
        config = BundleConfig.from_json(json.loads((bundle / "config.json").read_text()))
        vocab = Vocab.load(bundle / "vocab.json")
        states = []
        for f in sorted(bundle.glob("detector-seed*.safetensors"), key=lambda p: int(p.stem.split("seed")[1])):
            if verify and f.name not in manifest["files"]:
                raise BundleError(f"{bundle}: {f.name} is not listed in manifest.json")
            states.append(load_file(f))
        if not states:
            raise BundleError(f"{bundle}: no detector weights")
        npc = NpcModel.load(bundle) if config.npc_enabled else None
        report = bundle / "report"
        return cls(config=config, vocab=vocab, signature_map=pd.read_parquet(bundle / "signature_map.parquet"),
                   detector_states=states, postfilter=PostFilter.load(bundle / "postfilter.skops", vocab), npc=npc,
                   manifest=manifest, report_dir=report if report.is_dir() else None,
                   source=source or str(bundle.resolve()))

    # ── saving ───────────────────────────────────────────────────────────────
    def save_pretrained(self, out: str | os.PathLike, *, provenance: dict | None = None) -> Path:
        """Write the bundle; ``provenance`` (MIBiG, negatives, counts, tuned values, ...) goes into the manifest."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        files = ["config.json", "vocab.json", "signature_map.parquet", "postfilter.skops"]
        (out / "config.json").write_text(json.dumps(self.config.to_json(), indent=2))
        self.vocab.save(out / "vocab.json")
        sm = self.signature_map[SIGNATURE_MAP_COLUMNS].sort_values("signature").reset_index(drop=True)
        sm.to_parquet(out / "signature_map.parquet", index=False)
        for i, st in enumerate(self.detector_states):
            name = f"detector-seed{i}.safetensors"
            save_file({k: st[k].contiguous() for k in sorted(st)}, out / name,
                      metadata={"vocab_size": str(len(self.vocab)), "seed_index": str(i)})
            files.append(name)
        self.postfilter.save(out / "postfilter.skops")
        if self.npc is not None:
            files += self.npc.save(out)
        if self.report_dir is not None and Path(self.report_dir).resolve() != (out / "report").resolve():
            shutil.rmtree(out / "report", ignore_errors=True)  # replace, never merge with an older report
            shutil.copytree(self.report_dir, out / "report")
        if (out / "report").is_dir():
            files += sorted(str(p.relative_to(out)) for p in (out / "report").rglob("*") if p.is_file())
        manifest = self._manifest(provenance or {})
        manifest["files"] = {f: sha256_file(out / f) for f in files}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "README.md").write_text(model_card(manifest, self.config))
        self.manifest = manifest
        return out

    def _manifest(self, provenance: dict) -> dict:
        r = load_recipe()
        base = {k: v for k, v in self.manifest.items() if k not in ("files",)}
        base.update({
            "schema_version": SCHEMA_VERSION,
            "sanntis_version": __version__,
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ips": base.get("ips") or {"version": r.ips.version, "interpro": r.ips.interpro,
                                       "apps": r.ips.app_names},
            "n_seeds": len(self.detector_states),
            "vocab_size": len(self.vocab),
        })
        base.update(provenance)
        return base

    # ── runtime ──────────────────────────────────────────────────────────────
    def detectors(self, device: str | torch.device = "cpu") -> list[Detector]:
        if self._detectors is None:
            ms = []
            for st in self.detector_states:
                m = Detector(len(self.vocab), self.config.detector)
                m.load_state_dict(st)
                m.eval()
                ms.append(m)
            self._detectors = ms
        return [m.to(device) for m in self._detectors]


def verify_files(bundle: Path, manifest: dict) -> None:
    files = manifest.get("files") or {}
    if not files:
        raise BundleError(f"{bundle}: manifest lists no files")
    for rel, digest in files.items():
        p = bundle / rel
        if not p.is_file():
            raise BundleError(f"{bundle}: missing file {rel}")
        if sha256_file(p) != digest:
            raise BundleError(f"{bundle}: sha256 mismatch for {rel}")


def check_ips(manifest: dict, *, allow_mismatch: bool = False) -> None:
    """The bundle's IPS version / InterPro release / apps must match the installed recipe's pin."""
    r = load_recipe()
    ips = manifest.get("ips", {})
    want = {"version": r.ips.version, "interpro": r.ips.interpro, "apps": sorted(r.ips.app_names)}
    got = {"version": ips.get("version"), "interpro": ips.get("interpro"), "apps": sorted(ips.get("apps") or [])}
    if got != want:
        msg = f"bundle IPS {got} differs from the installed sanntis IPS pin {want}"
        if not allow_mismatch:
            raise IpsMismatchError(msg + " (use --allow-ips-mismatch to override)")
        log.warning(msg)


def _perf_table(perf: dict) -> list[str]:
    """``perf``: {dataset: {metric: value}} → markdown rows."""
    cols = [("precision_t", "Precision_T"), ("recall_macro_t", "Recall_macro_T"), ("f1_macro_t", "F1_macro_T"),
            ("bgc_precision", "BGC precision"), ("bgc_macro_recall", "BGC macro recall"), ("f2_macro", "F2_macro")]
    rows = ["| Dataset | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in perf.items():
        rows.append(f"| {name} | " + " | ".join(f"{m[k]:.3f}" if k in m else "–" for k, _ in cols) + " |")
    return rows


CITATION = """@article{sanchez2023sanntis,
  title   = {Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS},
  author  = {Sanchez, Santiago and Rogers, Joel D. and Rogers, Alexander B. and Nassar, Maaly and McEntyre, Johanna
             and Welch, Martin and Hollfelder, Florian and Finn, Robert D.},
  journal = {bioRxiv},
  year    = {2023},
  doi     = {10.1101/2023.05.23.540769}
}"""


def _floor_text(tuned: dict) -> str:
    labels = {"bgc_precision": "BGC precision", "precision_t": "Precision_T"}
    metric, floor = tuned.get("floor_metric"), tuned.get("floor")
    return f"{labels.get(metric, metric)} >= {floor:.2f}" if floor is not None else "a precision floor"


def model_card(manifest: dict, config: BundleConfig) -> str:
    ips = manifest.get("ips", {})
    mib = manifest.get("mibig", {})
    neg = (manifest.get("negatives") or {}).get("version", "")
    tuned = manifest.get("tuned", {}) or {}
    counts = manifest.get("counts", {}) or {}
    perf = manifest.get("performance", {}) or {}
    lines = [
        "---", "library_name: sanntis", "license: apache-2.0",
        "tags: [biology, genomics, biosynthetic-gene-clusters, secondary-metabolism, interproscan]", "---", "",
        "# SanntiS: biosynthetic gene cluster detection", "",
        "SanntiS detects biosynthetic gene clusters (BGCs) in prokaryotic and eukaryotic sequences from InterProScan "
        "domain annotations, assigns a MIBiG class and predicts NPClassifier pathways and superclasses of the "
        "products. This repository holds one model bundle: a 3-seed Conv1D + BiLSTM token detector, a RandomForest "
        "post-filter / MIBiG class head and an NPClassifier ensemble (RandomForest + logistic regression + kNN).", "",
        "## Use", "", "```bash",
        "sanntis setup --container-engine docker            # InterProScan 6 + InterPro data, once",
        "sanntis predict genome.fna --container-engine docker --model <this repo>@<revision>",
        "sanntis predict assembly.gbk --ips-tsv interproscan.tsv   # precomputed annotations, no containers",
        "```", "", "```python", "from sanntis import SanntisModel, Predictor",
        "model = SanntisModel.from_pretrained('<this repo>', revision='<revision>')", "```", "",
        "Inputs: prokaryotic nucleotide FASTA (genes called with pyrodigal), GenBank with CDS translations or a "
        "protein FASTA with the GFF3 of its CDS (prokaryotes and eukaryotes), each optionally with a precomputed "
        "InterProScan 5.76+ / 6.x TSV. Outputs: GFF3 (one `CLUSTER` feature per region with `bgc_class`, "
        "`p_non_bgc`, `npc_pathway`, `npc_superclass`, `npc_pathway_key_domains_unsupported`, ...), per-CDS scores, "
        "a regions table and antiSMASH sideloader JSON.", "",
        "## Model", "",
        "| Field | Value |", "|---|---|",
        f"| sanntis version | {manifest.get('sanntis_version', '')} |",
        f"| Fit | {manifest.get('fit', '')} |",
        f"| Features | InterProScan {ips.get('version', '')}, InterPro {ips.get('interpro', '')}; "
        f"{len(ips.get('apps', []))} member databases ({', '.join(ips.get('apps', []))}) |",
        f"| Vocabulary | {manifest.get('vocab_size', '')} tokens (incl. 3 specials) |",
        f"| Detector | {manifest.get('n_seeds', '')} seeds, {tuned.get('epochs', '')} epochs (median best CV epoch) |",
        f"| Operating point t\\* | {config.decode.threshold} (hysteresis low {config.decode.threshold_low}); "
        f"highest F2_macro on validation with {_floor_text(tuned)} |",
        f"| RF post-filter | drop regions with P(non-BGC) >= {config.postfilter_drop_threshold} |",
        f"| Heads | MIBiG class: {config.class_enabled}; NPClassifier: {config.npc_enabled} |",
        "", "## Training data", "",
        ("- Positives: " + (f"MIBiG {mib['release']} entries (status {', '.join(mib.get('statuses') or ['active'])})"
                            if mib.get("release") else "")
         + (" and " if mib.get("release") and manifest.get("external") else "")
         + (f"{sum(e.get('n', 0) for e in manifest['external'])} external BGCs (MIBiG format)"
            if manifest.get("external") else "")
         + ", deduplicated (composite Dice >= 0.974, longest kept); loci > 200 kb and loci without CDS excluded."),
        f"- Negatives: genome pool {neg} (GTDB r226 and RefSeq eukaryote representatives; native "
        "BGCs removed with antiSMASH 8.0.1): length-matched chunks and BGC-free background pieces.",
        "- Entities: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(counts.items())) + ".",
        "",
    ]
    if manifest.get("debug_max_epochs"):
        lines += ["> **Warning:** trained with `SANNTIS_DEBUG_MAX_EPOCHS="
                  f"{manifest['debug_max_epochs']}` (test / smoke run). Not a usable model.", ""]
    if perf:
        lines += ["## Performance", "",
                  "Metrics in ORF space. Validation = pooled out-of-fold CV predictions (training data).",
                  "", *_perf_table(perf), "",
                  "Full report with confidence intervals and figures: `report/report.md`.", ""]
    lines += [
        "## Limitations", "",
        "- Tokens come from InterProScan member-database hits; predictions on annotations from other InterProScan "
        "or InterPro releases drift (sanntis warns and reports coverage QC). Unannotated genes carry no signal.",
        "- Eukaryotic negatives cover fungi and protists only (no plants); plant BGC estimates are anecdotal.",
        "- BGC families with low similarity to MIBiG are recalled less well than near-duplicates (novelty analysis in "
        "the sanntisv1-data-and-analysis repository).",
        "- `p_non_bgc` and the NPClassifier scores are model scores, not calibrated probabilities (the NPC "
        "abstention tiers target 0.9 / 0.8 precision on validation).",
        "- `npc_pathway_key_domains_unsupported` is a hint, not a verdict: a called pathway none of whose key domains "
        "(or domain pairs) is in the region is wrong more often than a supported one, but most wrong calls do carry "
        "the domains (hybrids, tailoring); pathways with low rule coverage in CV are `na`.", "",
        "## Integrity", "", "Every file is listed with its sha256 in `manifest.json` and verified on load. "
        "No pickles: detectors are safetensors, sklearn models skops (explicit trusted types), the kNN reference a "
        "sparse npz.", "",
        "## Citation", "", "```bibtex", *CITATION.splitlines(), "```", "",
    ]
    return "\n".join(lines)
