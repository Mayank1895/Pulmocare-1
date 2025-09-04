from __future__ import annotations
from typing import Any, Dict, Optional, Union

from PIL import Image

from utils.model_manager import ModelManager, ValidationError, ArtifactError


def aggregate_predict(
    mm: ModelManager,
    image: Optional[Union[str, bytes, Image.Image]] = None,
    symptoms: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs pneumonia/tb/covid binary classifiers (image and/or symptoms).
    Averages modalities that are present. Chooses final label with per-disease threshold.
    """
    if image is None and symptoms is None:
        raise ValidationError("Provide at least one of: image, symptoms.")

    probs: Dict[str, float] = {}
    used_modalities = set()

    # ---------------- IMAGE MODELS ----------------
    if image is not None:
        for disease in mm.diseases:
            if disease in mm.xray_models:
                try:
                    p_img = mm.predict_image(image, disease)
                    probs[disease] = p_img
                    used_modalities.add("image")
                except ArtifactError:
                    # model not available for this disease — skip
                    pass

    # ---------------- SYMPTOM MODELS ----------------
    if symptoms is not None and mm.symptom_models:
        sym_res = mm.predict_symptoms(symptoms)  # returns {probabilities, primary_prediction, confidence}
        probs.update(sym_res["probabilities"])
        used_modalities.add("symptoms")

    if not probs:
        raise ValidationError("No predictions computed — check inputs and available models.")

    # ---------------- FINAL DECISION ----------------
    top_disease, top_prob = max(probs.items(), key=lambda kv: kv[1])

    # Default threshold
    threshold = float(mm.artifacts[top_disease].threshold) if top_disease in mm.artifacts else 0.5

    if top_disease == "normal":
        primary = "normal"
        confidence = probs.get("normal", 1.0)
    elif top_prob >= threshold:
        primary = top_disease
        confidence = top_prob
    else:
        primary = "normal"
        confidence = 1.0

    # Ensure "normal" probability always present
    if "normal" not in probs:
        probs["normal"] = max(0.0, 1.0 - max(probs.values()))

    return {
        "status": "success",
        "used_modalities": sorted(used_modalities),
        "predictions": probs,
        "primary_prediction": primary,
        "confidence": round(float(confidence), 6),
        "threshold_used": threshold,
    }
