from enum import Enum


class DType(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"
    DATE = "DATE"

    def __repr__(self) -> str:
        return f"DType.{self.name}"
        