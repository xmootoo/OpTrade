from typing import Optional
from rich.console import Console

# Custom error codes
MISSING_DATES = 1001
INCOMPATIBLE_START_DATE = 1002
INCOMPATIBLE_END_DATE = 1003
MARKET_HOLIDAY = 1004
WEEKEND = 1005
UNKNOWN_ERROR = 9999


class DataValidationError(Exception):
    def __init__(
        self,
        message: str,
        error_code: int,
        real_start_date: Optional[str] = None,
        real_end_date: Optional[str] = None,
        verbose: bool = True,
        warning: bool = False,
    ):

        self.message = message
        self.error_code = error_code
        self.real_start_date = real_start_date
        self.end_date = real_end_date

        error_dict = {
            INCOMPATIBLE_START_DATE: "Option data queried before contract start date",
            INCOMPATIBLE_END_DATE: "Option data queried before contract start date, and data ends before specified expirations",
            MISSING_DATES: "Option dates is a subset of stock dates, there are missing dates in between",
            MARKET_HOLIDAY: "Market holiday",
            WEEKEND: "Weekend",
            UNKNOWN_ERROR: "Unknown error",
        }

        error_str = error_dict[error_code]
        if verbose:
            ctx = Console()
            if warning:
                ctx.log(f"[bold yellow]WARNING ({error_str}):[/bold yellow] {message}")
            else:
                ctx = Console()
                ctx.log(f"[bold red]ERROR ({error_str}):[/bold red] {message}")

        # Standard exception behavior remains the same
        super().__init__()
