"""Write small launcher and provenance JSON records."""

import json
from pathlib import Path


def atomic(path, value):
    """Replace a JSON file through a temporary sibling on the same filesystem.

    Parent directories are created. Existing destinations are replaced; callers
    must serialize writes to the same path because the temporary name is fixed.
    This helper does not fsync or provide a no-overwrite publication guarantee.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)
