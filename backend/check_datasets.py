import logging
logging.disable(logging.CRITICAL)
import sys
sys.path.insert(0, '/app')
from app.core.database import SyncSessionLocal
from app.models.dataset import Dataset

s = SyncSessionLocal()
rows = s.query(Dataset.name, Dataset.status, Dataset.dataset_type, Dataset.parent_id).order_by(Dataset.created_at.desc()).limit(20).all()
print(f"{'Name':<45} {'Status':<15} {'Type':<12} {'Has Parent'}")
print("-" * 90)
for r in rows:
    print(f"{str(r.name):<45} {str(r.status):<15} {str(r.dataset_type):<12} {r.parent_id is not None}")
s.close()
