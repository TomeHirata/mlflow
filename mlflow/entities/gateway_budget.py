import calendar
import datetime
from dataclasses import dataclass


def _compute_period_dates(renewal_period: str, created_at_ms: int) -> tuple[str, str]:
    """
    Compute the start and end dates of the current budget window based on the renewal period.

    Args:
        renewal_period: One of "monthly", "quarterly", or "annually".
        created_at_ms: Budget creation timestamp in milliseconds (used to anchor quarterly periods).

    Returns:
        A tuple of (period_start, period_end) as ISO 8601 date strings (YYYY-MM-DD).
    """
    today = datetime.date.today()

    if renewal_period == "monthly":
        period_start = today.replace(day=1)
        last_day = calendar.monthrange(today.year, today.month)[1]
        period_end = today.replace(day=last_day)
    elif renewal_period == "quarterly":
        quarter = (today.month - 1) // 3
        quarter_start_month = quarter * 3 + 1
        quarter_end_month = quarter_start_month + 2
        last_day = calendar.monthrange(today.year, quarter_end_month)[1]
        period_start = today.replace(month=quarter_start_month, day=1)
        period_end = today.replace(month=quarter_end_month, day=last_day)
    else:
        # annually
        period_start = today.replace(month=1, day=1)
        period_end = today.replace(month=12, day=31)

    return period_start.isoformat(), period_end.isoformat()


@dataclass
class GatewayBudget:
    budget_id: str
    name: str
    amount: float
    currency: str
    renewal_period: str
    current_spending: float
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None

    @property
    def period_start(self) -> str:
        start, _ = _compute_period_dates(self.renewal_period, self.created_at)
        return start

    @property
    def period_end(self) -> str:
        _, end = _compute_period_dates(self.renewal_period, self.created_at)
        return end

    def to_dict(self) -> dict:
        return {
            "budget_id": self.budget_id,
            "name": self.name,
            "amount": self.amount,
            "currency": self.currency,
            "renewal_period": self.renewal_period,
            "current_spending": self.current_spending,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
        }
