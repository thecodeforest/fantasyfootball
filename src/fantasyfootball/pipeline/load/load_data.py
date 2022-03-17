import shutil
from pathlib import Path

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.pipeline_logger import logger
from fantasyfootball.pipeline.utils import concat_ff_csv, read_args


def copy_to_git_data_dir(root_dir: str, season_year: int) -> None:
    """Copies processed data to the git data directory, which is the
       data surfaced to the end-user.

    Args:
        root_dir (str): The root directory of the data pipeline.
        season_year (int): The season year to copy the data for.

    Raises:
        Exception: If directory is empty and no data is present, an error is raised.

    Returns:
        None
    """
    dir_suffix = Path("data") / "season" / str(season_year)
    processed_source_dir = Path(root_dir) / dir_suffix / "processed"
    processed_dest_dir = Path(root_dir).parent.parent / dir_suffix
    processed_data_sources = list(processed_source_dir.glob("*"))
    data_source_types = [str(x).split("/")[-1] for x in processed_data_sources]
    for source_type in data_source_types:
        source_type_fname = source_type + ".csv"
        all_file_paths = list((processed_source_dir / source_type).glob("*.csv"))
        if len(all_file_paths) == 0:
            raise Exception(f"No files found in {source_type}")
        if len(all_file_paths) == 1:
            shutil.copy(
                src=all_file_paths[0], dst=processed_dest_dir / source_type_fname
            )
        if len(all_file_paths) > 1:
            df = concat_ff_csv(file_paths=all_file_paths)
            df.to_csv(processed_dest_dir / source_type_fname, index=False)
        logger.info(f"copying {source_type} data")
    return None


if __name__ == "__main__":
    args = read_args()
    copy_to_git_data_dir(root_dir=root_dir, season_year=args.season_year)
