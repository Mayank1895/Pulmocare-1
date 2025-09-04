from __future__ import annotations
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from pydantic import BaseModel, Field
from services.predictor import aggregate_predict
from utils.model_manager import ModelManager, ValidationError, ArtifactError


router = APIRouter(prefix="/predict", tags=["predict"])

# We initialize a single ModelManager at import time so it loads once.
mm = ModelManager()
mm.load_all()


# ---------- Schemas ----------
class SymptomPayload(BaseModel):
    symptoms: Dict[str, Any] = Field(default_factory=dict, description="Symptom key/value pairs")


class CombinedPayload(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image (data URL or raw)")
    symptoms: Optional[Dict[str, Any]] = None


# ---------- Endpoints ----------

@router.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(mm.device),
        "diseases": mm.diseases,
        "symptom_models": list(mm.symptom_models.keys()),
        "xray_models": list(mm.xray_models.keys()),
        "artifacts": list(mm.artifacts.keys()),
    }


@router.post("/image")
async def predict_from_image(file: UploadFile = File(..., description="X-ray image file (jpg/png)")):
    try:
        data = await file.read()
        result = aggregate_predict(mm, image=data, symptoms=None)
        return result  # ✅ don’t override predicted
    except (ValidationError, ArtifactError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")


@router.post("/symptoms")
def predict_from_symptoms(payload: SymptomPayload):
    try:
        result = aggregate_predict(mm, image=None, symptoms=payload.symptoms)
        return result  # ✅ don’t override predicted
    except (ValidationError, ArtifactError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")


@router.post("/combined")
def predict_combined(payload: CombinedPayload = Body(...)):
    try:
        image = payload.image_base64
        symptoms = payload.symptoms
        if image is None and not symptoms:
            raise HTTPException(status_code=400, detail="Provide at least image_base64 or symptoms.")
        result = aggregate_predict(mm, image=image, symptoms=symptoms)
        return result  # ✅ don’t override predicted
    except (ValidationError, ArtifactError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")


@router.post("/multipart")
async def predict_multipart(file: Optional[UploadFile] = File(None), symptoms_json: Optional[str] = None):
    try:
        image_bytes = await file.read() if file is not None else None
        symptoms = json.loads(symptoms_json) if symptoms_json else None
        if image_bytes is None and not symptoms:
            raise HTTPException(status_code=400, detail="Provide file and/or symptoms_json.")
        result = aggregate_predict(mm, image=image_bytes, symptoms=symptoms)
        return result  # ✅ don’t override predicted
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid symptoms_json (must be valid JSON).")
    except (ValidationError, ArtifactError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")
