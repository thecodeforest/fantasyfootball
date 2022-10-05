import sys
import pandas as pd
import awswrangler as wr
from pathlib import Path
from collections import Counter

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import read_args  # noqa: E402


def fetch_all_staging_data_paths(base_path: Path) -> list:
    """Fetch all staging data paths

    Args:
        base_path (Path): Base path to start searching from

    Returns:
        list: List of all staging data paths

    Examples:
        >>> fetch_all_staging_data_paths(Path('fantasyfootball/staging_datasets/season/2022'))
        [PosixPath('fantasyfootball/staging_datasets/season/2022/processed/draft/draft.csv'),]
    """  # noqa: E501
    # list all directories and subdirectories recursively
    paths = list(base_path.glob("**/*"))
    # filter only to directories
    paths = [path for path in paths if path.is_dir()]
    # calculate the length of each path
    path_lengths = [len(path.parts) for path in paths]
    # get a count of each path legnth
    most_common_length = Counter(path_lengths).most_common(1)[0][0]
    # filter to only the most common path length
    paths = [path for path in paths if len(path.parts) == most_common_length]
    return paths


def _extract_path_parts(path: Path, season_year: int) -> tuple:
    """Extracts all parts of path after the season year

    Args:
        path (Path): Path to extract parts from
        season_year (int): Season year to start extracting from

    Returns:
        tuple: Tuple of path parts

    Examples:
        >>> _extract_path_parts(Path('fantasyfootball/staging_datasets/season/2022/processed/stats/WillJa10_stats.csv'), 2022)
        ('processed', 'stats', 'WillJa10_stats.csv')
    """  # noqa: E501
    # find the index of path that contains the seasonyear
    season_year_index = path.parts.index(str(season_year))
    # filter only to the path parts after the season_year_index
    path_parts = path.parts[season_year_index:]
    return path_parts


def create_io_paths(path: Path, s3_path_prefix: str, season_year: int) -> tuple:
    """Create the input and output paths for a given path.

    Args:
        path (Path): Path to create input and output paths for
        s3_path_prefix (str): S3 path prefix to use
        season_year (int): Season year to use

    Returns:
        tuple: Tuple of input and output paths

    Examples:
        >>> create_io_paths(Path('fantasyfootball/staging_datasets/season/2022/processed/draft/draft.csv'), 'fantasy-football-pipeline/datasets', 2022)
        (PosixPath('fantasyfootball/staging_datasets/season/2022/processed/draft/draft.csv'), 'fantasy-football-pipeline/datasets/season/2022/processed/draft/draft.csv')
    """  # noqa: E501
    # list all the .csv files in the directory
    local_file_paths = list(path.glob("*.csv"))
    all_path_parts = [_extract_path_parts(x, season_year) for x in local_file_paths]
    s3_paths = [s3_path_prefix + "/".join(x) for x in all_path_parts]
    return (local_file_paths, s3_paths)


if __name__ == "__main__":
    args = read_args()
    source_dir = Path(root_dir) / "staging_datasets" / "season" / str(args.season_year)
    s3_path_prefix = f"s3://{args.s3_bucket}/datasets/season/"
    paths = fetch_all_staging_data_paths(source_dir)
    for path in paths:
        logger.info(f"Saving {path} to S3")
        input_local_file_path, output_s3_paths = create_io_paths(
            path, s3_path_prefix, args.season_year
        )
        for input_path, output_path in zip(input_local_file_path, output_s3_paths):
            wr.s3.to_csv(df=pd.read_csv(input_path), path=output_path, index=False)
