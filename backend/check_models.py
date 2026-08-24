import logging
logging.disable(logging.CRITICAL)
import sys
sys.path.insert(0, '/app')
from app.core.database import SyncSessionLocal
from app.models.model import Model
from app.models.training import TrainingJob

s = SyncSessionLocal()
models = s.query(Model.name, Model.version, Model.status).all()
print("--- MODELS ---")
for m in models:
    print(m)

jobs = s.query(TrainingJob.name, TrainingJob.status).order_by(TrainingJob.created_at.desc()).limit(5).all()
print("\n--- RECENT TRAINING JOBS ---")
for j in jobs:
    print(j)
s.close()
