import logging
from datetime import datetime
from pathlib import Path

import sys

sys.path.append(str(Path.cwd()))
from pipeline.utils import create_dir  # noqa: E402

pipeline_log_dir = Path(__file__).parent / "logs"
create_dir(pipeline_log_dir)

logger = logging
logger.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.INFO,
    filename="{log_dir}/pipeline-run-{start_time}.log".format(
        log_dir=str(pipeline_log_dir), start_time=datetime.now().strftime("%Y-%m-%d")
    ),
)
