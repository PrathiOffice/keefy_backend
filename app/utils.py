import os
from typing import Optional, Tuple
from starlette.responses import StreamingResponse, Response

CHUNK_SIZE = 1024 * 1024  # 1MB

def open_file_range(path: str, range_header: Optional[str]) -> Tuple[Response, int]:
    file_size = os.path.getsize(path)
    if range_header is None:
        def iterfile():
            with open(path, "rb") as f:
                while chunk := f.re