from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- import CORS middleware
from routers.predict import router as predict_router

app = FastAPI(title="PulmoCare Backend", version="1.0.0")

# ---------------- CORS Setup ----------------
# For development, allow all origins. Replace '*' with your frontend URL in production
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # allowed origins
    allow_credentials=True,         # allow cookies
    allow_methods=["*"],            # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # allow all headers
)
# -------------------------------------------

# Include your routers
app.include_router(predict_router)

# Root endpoint
@app.get("/")
def root():
    return {"status": "ok", "message": "PulmoCare backend is running"}
