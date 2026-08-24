import pytest

from app.api.v1.dashboard import get_dashboard_stats


class _FakeUser:
    def __init__(self, user_id: str):
        self.id = user_id


class _FakeDB:
    def __init__(self, scalar_values):
        self._values = iter(scalar_values)

    async def scalar(self, _query):
        return next(self._values)


@pytest.mark.asyncio
async def test_dashboard_stats_returns_aggregated_counts():
    db = _FakeDB([3, 10, 2, 4, 1])
    user = _FakeUser("0f8fad5b-d9cb-469f-a165-70867728950e")

    result = await get_dashboard_stats(db=db, current_user=user)

    assert result["data"] == {
        "total_datasets": 3,
        "total_training_jobs": 10,
        "active_training_jobs": 2,
        "production_models": 4,
        "active_alerts": 1,
    }

