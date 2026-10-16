# Models directory

Saved fitted `sklearn` pipelines (`preprocessing + classifier`) via joblib.

Expected files after training (git-ignored except this README):
- `pipeline_binary.joblib` — binary benign vs attack
- `pipeline_multiclass.joblib` — benign / light / heavy (only if labels support it)
- Sidecar JSONs: `pipeline_binary.meta.json`, `pipeline_multiclass.meta.json`
  containing feature names, order, label mapping, config, metrics, versions.

Each `.joblib` is a dict: `{"pipeline": Pipeline, "feature_names": [...], "label_classes": [...], "task": "binary"|"multiclass", "metrics": {...}, "config_snapshot": {...}}`

Load:
```python
from src.predict import load_artifact
art = load_artifact("models/pipeline_binary.joblib")
print(art["feature_names"], art["label_classes"])
```

Never save an unfitted scaler/encoder. Always save the fitted pipeline from training.
Synthetic-demo pipelines are prefixed `synthetic_` and must not be presented as CIC-trained.
