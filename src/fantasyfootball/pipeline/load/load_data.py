import gzip
import os
import shutil
from pathlib import Path, PosixPath

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.pipeline_logger import logger
from fantasyfootball.pipeline.utils import concat_ff_csv, create_dir, read_args


def copy_to_git_data_dir(source_dir: PosixPath, dest_dir: PosixPath) -> None:
    """Copies processed data to the git data directory, which is the
       data surfaced to the end-user.

    Args:
        source_dir (PosixPath): The source directory to copy the data from.
        dest_dir (PosixPath): The destination directory to copy the data to.

    Raises:
        Exception: If directory is empty and no data is present, an error is raised.

    Returns:
        None
    """
    create_dir(dest_dir)
    processed_data_sources = list(source_dir.glob("*"))
    data_source_types = [
        x
        for x in [str(x).split("/")[-1] for x in processed_data_sources]
        if x != ".DS_Store"
    ]
    for source_type in data_source_types:
        source_type_fname = source_type + ".csv"
        all_file_paths = list((source_dir / source_type).glob("*.csv"))
        if len(all_file_paths) == 0:
            raise Exception(f"No files found in {source_type}")
        if len(all_file_paths) == 1:
            shutil.copy(src=all_file_paths[0], dst=dest_dir / source_type_fname)
        if len(all_file_paths) > 1:
            df = concat_ff_csv(file_paths=all_file_paths)
            df.to_csv(dest_dir / source_type_fname, index=False)
        logger.info(f"copying {source_type} data")
    return None


def zip_git_data_dir(dest_dir: PosixPath) -> None:
    """Zips all the .csv files in the git data directory.

    Args:
        dest_dir (PosixPath): The destination directory to zip the data to.

    Returns:
        None
    """
    for dir_name, sub_dir_name, file_names in os.walk(dest_dir):
        for file_name in file_names:
            file_path = "/".join([dir_name, file_name])
            if Path(file_path).suffix == ".csv":
                with open(file_path, "rb") as csv_in:
                    with gzip.open(file_path.replace(".csv", ".gz"), "wb") as gzip_out:
                        shutil.copyfileobj(csv_in, gzip_out)
                        logger.info(f"compressing {file_path}")
    return None


if __name__ == "__main__":
    args = read_args()
    dir_suffix = Path("datasets") / "season" / str(args.season_year)
    source_dir = Path(root_dir) / dir_suffix / "processed"
    dest_dir = Path(root_dir).parent.parent / dir_suffix
    copy_to_git_data_dir(source_dir=source_dir, dest_dir=dest_dir)
    zip_git_data_dir(dest_dir=dest_dir)
