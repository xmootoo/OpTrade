from typing import Optional
from rich.console import Console

# Custom error codes
MISSING_DATES = 1001
OPTION_DATE_MISMATCH = 1002

class DataValidationError(Exception):
    def __init__(self, message: str, error_code: int, data_str: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        self.data_str = data_str

        error_dict = {MISSING_DATES: "Missing dates", OPTION_DATE_MISMATCH: "Option date mismatch"}
        error_str = error_dict[error_code]

        # Log using ctx
        ctx = Console()
        ctx.log(f"[bold red]ERROR ({error_str}):[/bold red] {message}")

        # Standard exception behavior remains the same
        super().__init__()
