from typing import Optional

# Custom error codes
MISSING_DATES = 1001
OPTION_DATE_MISMATCH = 1002

# Define a custom exception class with both error code and data string
class DataValidationError(Exception):
    def __init__(self, message: str, error_code:int , data_str=Optional[str]):
        self.message = message
        self.error_code = error_code
        self.data_str = data_str

        error_dict = {MISSING_DATES: "Missing dates", OPTION_DATE_MISMATCH: "Option date mismatch"}
        error_str = error_dict[error_code]

        # Combine for string representation
        super().__init__(f"[Error {error_str}] {message}")
