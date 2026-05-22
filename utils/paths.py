from datetime import datetime
from pathlib import Path


def create_run_dir(
    root: str = "results",
) -> Path:
    """
    Create a unique run directory.

    Example:
    results/2026-05-22_14-35-10/
    """

    run_id = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_dir = Path(root) / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir