from __future__ import annotations
import base64
import io
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn


# ---------- Errors ----------
class ArtifactError(RuntimeError):
    pass


class ValidationError(ValueError):
    pass


# ---------- Helpers ----------
def _safe_isnan(x: Any) -> bool:
    return isinstance(x, (float, int, np.floating, np.integer)) and np.isnan(x)  # type: ignore[arg-type]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class DiseaseArtifacts:
    feature_order: List[str]
    impute_stats: Dict[str, Any]
    col_info: Dict[str, Any]
    column_mapping: Optional[Dict[str, str]]
    threshold: float
    image_norm_mean: Tuple[float, float, float]
    image_norm_std: Tuple[float, float, float]
    image_size: Tuple[int, int]


class ModelManager:
    """
    Handles loading + inference for:
      - X-ray torch models (.pt / .pth)
      - Symptom models (scikit-learn pipelines via joblib)
      - Per-disease artifacts
    """

    def __init__(self) -> None:
        # project root
        self.ROOT = Path(__file__).resolve().parents[2]
        self.MODELS_DIR = self.ROOT / "models"
        self.ARTIFACTS_DIR = self.ROOT / "artifacts"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diseases = ["pneumonia", "tb", "covid"]

        self.symptom_models: Dict[str, Any] = {}
        self.xray_models: Dict[str, Any] = {}
        self.artifacts: Dict[str, DiseaseArtifacts] = {}
        self.target_encoders: Dict[str, Any] = {}

    # ---------- Public API ----------
    def load_all(self) -> None:
        self._load_symptom_models()
        self._load_xray_models()
        self._load_artifacts()
        print("[ModelManager] Loaded:",
              "\n  symptom:", list(self.symptom_models.keys()),
              "\n  xray:", list(self.xray_models.keys()),
              "\n  artifacts:", list(self.artifacts.keys()),
              "\n  encoders:", list(self.target_encoders.keys()))

    # ---------- Loading internals ----------
    def _load_symptom_models(self) -> None:
        sdir = self.MODELS_DIR / "symptom"
        if not sdir.is_dir():
            raise ArtifactError(f"Missing directory: {sdir}")

        for disease in self.diseases:
            candidates = list(sdir.glob(f"*{disease}*symptom*model*.pkl"))
            if not candidates:
                candidates = list(sdir.glob(f"{disease}_symptom_model.pkl"))
            if not candidates:
                raise ArtifactError(f"Symptom model not found for '{disease}' in {sdir}")
            path = sorted(candidates)[0]
            self.symptom_models[disease] = joblib.load(path)

    def _load_xray_models(self) -> None:
        xdir = self.MODELS_DIR / "xray"
        if not xdir.is_dir():
            print(f"[ModelManager] Warning: no xray dir at {xdir} — image predictions disabled.")
            return

        mapping = {
            "covid": ["covid", "image", "xray"],
            "pneumonia": ["pneumonia", "xray", "image"],
            "tb": ["tb", "tuberculosis", "image", "xray"],
        }

        for disease in self.diseases:
            cand: List[Path] = []
            for ext in (".pth", ".pt"):
                for token in mapping[disease]:
                    cand.extend([p for p in xdir.glob("**/*") if token in p.stem.lower() and p.suffix.lower() == ext])

            seen = set()
            uniq = [p for p in cand if not (p in seen or seen.add(p))]
            if not uniq:
                print(f"[ModelManager] Info: no xray model file detected for '{disease}' in {xdir}.")
                continue

            path = uniq[0]
            model = self._try_load_torch_model(path)
            self.xray_models[disease] = model

    def _try_load_torch_model(self, path: Path) -> Any:
        try:
            m = torch.jit.load(str(path), map_location=self.device)
            m.eval()
            return m
        except Exception:
            pass
        # fallback to pickled nn.Module
        obj = torch.load(str(path), map_location=self.device, weights_only=False)
        if hasattr(obj, "eval") and callable(getattr(obj, "eval")):
            obj.eval()
            return obj
        raise ArtifactError(f"X-ray model at {path} appears to be a state_dict. Save full nn.Module or TorchScript.")

    def _load_artifacts(self) -> None:
        if not self.ARTIFACTS_DIR.is_dir():
            raise ArtifactError(f"Missing artifacts dir: {self.ARTIFACTS_DIR}")

        for disease in self.diseases:
            ddir = self.ARTIFACTS_DIR / disease
            if not ddir.is_dir():
                raise ArtifactError(f"Missing disease artifacts folder: {ddir}")

            fo = self._first_existing(ddir, [f"{disease}_feature_order.json", "feature_order.json"])
            im = self._first_existing(ddir, [f"{disease}_impute_stats.json", "impute_stats.json"])
            ci = self._first_existing(ddir, [f"{disease}_col_info.json", "col_info.json"])
            cm = self._first_existing(ddir, [f"{disease}_symptoms_column_mapping.json"], required=False)
            te = self._first_existing(ddir, [f"{disease}_target_encoder.pkl"], required=False)

            feature_order = _read_json(fo)
            impute_stats = _read_json(im)
            col_info = _read_json(ci)
            column_mapping = _read_json(cm) if cm else None
            if te:
                self.target_encoders[disease] = joblib.load(te)

            threshold = float(col_info.get("threshold", 0.30))
            img_cfg = col_info.get("image_normalization", {})
            image_size = tuple(img_cfg.get("size", (224, 224))) if isinstance(img_cfg, dict) else (224, 224)
            mean = tuple(img_cfg.get("mean", (0.485, 0.456, 0.406))) if isinstance(img_cfg, dict) else (0.485, 0.456, 0.406)
            std = tuple(img_cfg.get("std", (0.229, 0.224, 0.225))) if isinstance(img_cfg, dict) else (0.229, 0.224, 0.225)

            self.artifacts[disease] = DiseaseArtifacts(
                feature_order=list(feature_order),
                impute_stats=dict(impute_stats),
                col_info=dict(col_info),
                column_mapping=column_mapping if isinstance(column_mapping, dict) else None,
                threshold=threshold,
                image_size=(int(image_size[0]), int(image_size[1])),
                image_norm_mean=(float(mean[0]), float(mean[1]), float(mean[2])),
                image_norm_std=(float(std[0]), float(std[1]), float(std[2])),
            )

    def _first_existing(self, folder: Path, names: List[str], required: bool = True) -> Optional[Path]:
        for n in names:
            p = folder / n
            if p.exists():
                return p
        if required:
            raise ArtifactError(f"Missing artifacts: tried {names} in {folder}")
        return None

    # ---------- Image preprocessing/inference ----------
    def _image_transform_for(self, disease: str) -> transforms.Compose:
        art = self.artifacts[disease]
        return transforms.Compose([
            transforms.Resize(art.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(art.image_norm_mean), std=list(art.image_norm_std)),
        ])

    @staticmethod
    def decode_image(image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            if "base64," in image_input:
                image_input = image_input.split("base64,")[-1]
            img = Image.open(io.BytesIO(base64.b64decode(image_input)))
        else:
            raise ValidationError("Unsupported image type.")
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def preprocess_image(self, image_input: Union[str, bytes, Image.Image], disease: str) -> torch.Tensor:
        if disease not in self.artifacts:
            raise ValidationError(f"Unknown disease for image preprocessing: {disease}")
        tfm = self._image_transform_for(disease)
        img = self.decode_image(image_input)
        return tfm(img).unsqueeze(0).to(self.device)

    def predict_image(self, image_input: Union[str, bytes, Image.Image], disease: str) -> float:
        if disease not in self.xray_models:
            raise ArtifactError(f"No X-ray model loaded for '{disease}'")
        x = self.preprocess_image(image_input, disease)
        model = self.xray_models[disease]
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.squeeze()
            prob = torch.sigmoid(out).float().mean().item()
        return float(max(0.0, min(1.0, prob)))

    # ---------- Symptom preprocessing ----------
    def process_symptoms(self, symptoms: Dict[str, Any], disease: str) -> np.ndarray:
        if disease not in self.artifacts:
            raise ValidationError(f"Unknown disease for symptom processing: {disease}")
        art = self.artifacts[disease]
        feats = list(art.feature_order)
        src = dict(symptoms or {})
        working: Dict[str, Any] = {}
        if art.column_mapping:
            for user_key, model_key in art.column_mapping.items():
                if user_key in src:
                    working[model_key] = src[user_key]
        for k, v in src.items():
            if k in feats and k not in working:
                working[k] = v

        ordered: List[float] = []
        for f in feats:
            val = working.get(f, None)
            if val is None or _safe_isnan(val):
                val = art.impute_stats.get(f, 0)
            if isinstance(val, bool):
                val = int(val)
            if isinstance(val, str):
                mapped = self._map_category(disease, f, val)
                if mapped is None:
                    raise ValidationError(f"Feature '{f}' has value '{val}', no mapping found.")
                val = mapped
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise ValidationError(f"Feature '{f}' must be numeric after processing; got {type(val)}")
            ordered.append(float(val))
        return np.asarray(ordered, dtype=np.float32).reshape(1, -1)

    def _map_category(self, disease: str, feature: str, value: str) -> Optional[Union[int, float]]:
        ci = self.artifacts[disease].col_info
        for key in ("categorical_maps", "encoders"):
            blob = ci.get(key)
            if isinstance(blob, dict):
                fmap = blob.get(feature)
                if isinstance(fmap, dict) and value in fmap:
                    return fmap[value]
        return None

    # ---------- Symptom inference ----------
    def predict_symptoms(self, symptoms: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
 
        results = {}

        # Run each model
        for disease, model in self.symptom_models.items():
            x = self.process_symptoms(symptoms, disease)
            p = float(model.predict_proba(x)[0][1])
            results[disease] = p

        # Pick best
        best_disease = max(results, key=results.get)
        best_prob = results[best_disease]

        # Apply fallback to normal
        if best_prob < threshold:
            return {
                "prediction": "normal",
                "confidence": 1.0,
                "probabilities": results
            }
        else:
            return {
                "prediction": best_disease,
                "confidence": best_prob,
                "probabilities": results
            }
