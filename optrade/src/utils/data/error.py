from typing import Optional
from rich.console import Console

# Custom error codes
MISSING_DATES = 1001
INCOMPATIBLE_START_DATE = 1002
INCOMPATIBLE_END_DATE = 1003
UNKNOWN_ERROR = 9999

class DataValidationError(Exception):
    def __init__(
        self,
        message: str,
        error_code: int,
        real_start_date: Optional[str] = None,
        real_end_date: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        self.real_start_date = real_start_date
        self.end_date = real_end_date

        error_dict = {
            MISSING_DATES: "Missing dates",
            INCOMPATIBLE_START_DATE: "Option start_date mismatch",
            INCOMPATIBLE_END_DATE: "Option start_date and end_date mismatch"}
        error_str = error_dict[error_code]

        # Log using ctx
        ctx = Console()
        ctx.log(f"[bold red]ERROR ({error_str}):[/bold red] {message}")

        # Standard exception behavior remains the same
        super().__init__()
