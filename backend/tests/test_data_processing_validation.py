import pytest
import pandas as pd
import io
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.data_processing_service import DataProcessingService

def test_prepare_training_data_empty_input():
    async def run_test():
        # Setup
        mock_db = AsyncMock()
        service = DataProcessingService(mock_db)
        
        # Mock mocks
        service.data_service.get_dataset = AsyncMock()
        service.storage.download_dataset = AsyncMock()
        
        # Mock DB return
        mock_dataset = MagicMock()
        mock_dataset.id = "test-id"
        mock_dataset.storage_path = "test/path"
        mock_dataset.file_format = "parquet"
        mock_dataset.name = "test_dataset"
        service.data_service.get_dataset.return_value = mock_dataset
        
        # Mock Empty Parquet Content
        df_empty = pd.DataFrame()
        buffer = io.BytesIO()
        df_empty.to_parquet(buffer, engine='pyarrow')
        service.storage.download_dataset.return_value = buffer.getvalue()
        
        # Execution & Assert
        await service.prepare_training_data("test-id", {})

    with pytest.raises(ValueError, match="Input dataset is empty"):
        asyncio.run(run_test())

def test_prepare_training_data_empty_split():
    async def run_test():
        # Setup
        mock_db = AsyncMock()
        service = DataProcessingService(mock_db)
        
        # Mock mocks
        service.data_service.get_dataset = AsyncMock()
        service.storage.upload_dataset = AsyncMock()
        service.storage.download_dataset = AsyncMock()
        
        # Mock DB return
        mock_dataset = MagicMock()
        mock_dataset.id = "test-id-2"
        mock_dataset.storage_path = "test/path/2"
        mock_dataset.file_format = "parquet"
        mock_dataset.version = "v1"
        mock_dataset.name = "test_dataset_2"
        service.data_service.get_dataset.return_value = mock_dataset
        
        # Mock Valid Data
        df_valid = pd.DataFrame({'col': [1, 2, 3]})
        buffer = io.BytesIO()
        df_valid.to_parquet(buffer, engine='pyarrow')
        service.storage.download_dataset.return_value = buffer.getvalue()
        
        # Patch train_test_split to return empty df
        with patch('app.services.data_processing_service.train_test_split') as mock_split:
            # Return (Valid Train, Empty Test)
            mock_split.return_value = (df_valid, pd.DataFrame())
            
            # Execution
            await service.prepare_training_data("test-id-2", {}, test_size=0.5)

    with pytest.raises(ValueError, match="Split resulted in empty dataframe"):
        asyncio.run(run_test())

