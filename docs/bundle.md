# Model bundle format

A model is one directory that holds every stage. It is loaded the same way from Hugging Face Hub or from a local
path:

```python
from sanntis import Predictor, SanntisModel

m = SanntisModel.from_pretrained("microbiome-informatics/sanntis@v1.0.0-beta.1")  # Hub, pinned revision
m = SanntisModel.from_pretrained("runs/my_retrain/bundle")            # local directory
pred = Predictor(m).predict(cds_table)  # columns record_id, cds_idx, locus_tag, start, end, tokens
```

| File | Content |
|---|---|
| `config.json` | detector, feature and decode settings, including the operating point t\*; RF drop threshold; classes; heads enabled; `coverage_reference` |
| `vocab.json` | 3 special tokens and the domain tokens |
| `signature_map.parquet` | mapping from member database signature to InterPro entry at the InterPro release of the model |
| `detector-seed{i}.safetensors` | detector ensemble weights |
| `postfilter.skops` | RandomForest post-filter and MIBiG class head |
| `npc.skops`, `npc_knn.npz`, `npc_thresholds.json` | NPClassifier head: members, kNN reference, pathway thresholds and abstention tiers. Absent when the head is disabled |
| `npc_key_domains.json` | NPC pathway key-domain rules: setting, status per pathway (`active`, `gated` or `no_rules`), out-of-fold coverage, rules with their counts. Optional; without it the key-domain columns are empty |
| `manifest.json` | sanntis version, InterProScan version, InterPro release, applications, provenance, sha256 of every file |
| `README.md` | model card generated from the manifest |
| `report/` | validation report of the training run |

Detector weights are stored as safetensors files. The scikit-learn models are stored as skops files and loaded
with an explicit list of trusted types.

At load time sanntis checks the sha256 of every file, the bundle schema version and the InterProScan version of
the model. `manifest.ips` must equal the version installed with sanntis; `--allow-ips-mismatch` turns the error
into a warning. A model that fails these checks is refused, so loading a model with `sanntis predict --model`
also verifies it.
