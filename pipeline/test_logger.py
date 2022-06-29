import sys
from pathlib import Path
import re

# import time
from itertools import chain, product
from typing import List, Tuple

import pandas as pd

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, stats_url  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402

if __name__ == "__main__":
    print("log some things")
    logger.info("test test logger")
    logger.info("test test test logger")
