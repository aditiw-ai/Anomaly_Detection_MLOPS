from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))


def now_ist():
    """Current time in IST. Returns naive datetime for DB compatibility with TIMESTAMP WITHOUT TIME ZONE."""
    return datetime.now(IST).replace(tzinfo=None)
