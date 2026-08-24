import asyncio
import uuid
import sys

async def main():
    try:
        from app.core.database import SessionLocal
        from app.services.feature_service import FeatureService
        
        db = SessionLocal()
        svc = FeatureService(db)
        try:
            res = await svc.create_feature_set(
                dataset_id=str(uuid.uuid4()), 
                name='Test', 
                config={}
            )
            print('SUCCESS')
        except Exception as e:
            print('ERROR TYPE:', type(e))
            print('ERROR:', repr(e))
    except ImportError as ie:
        print("Import Error:", ie)

if __name__ == "__main__":
    asyncio.run(main())
