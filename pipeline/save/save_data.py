import sys
import pandas as pd
import awswrangler as wr
from pathlib import Path

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, pipeline_output_s3_bucket  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import read_args  # noqa: E402


if __name__ == "__main__":
    args = read_args()
    source_dir = (
        Path(root_dir)
        / "src"
        / "fantasyfootball"
        / "datasets"
        / "season"
        / str(args.season_year)
    )
    all_file_paths = list(source_dir.glob("*.gz"))
    file_names = [x.stem for x in all_file_paths]
    for file_path, file_name in zip(all_file_paths, file_names):
        df = pd.read_csv(file_path, compression="gzip")
        s3_output_path = f"s3://{pipeline_output_s3_bucket}/datasets/season/{args.season_year}/{file_name}.csv"  # noqa: E501
        wr.s3.to_csv(df, s3_output_path, index=False)
        logger.info(f"Saving {file_name} to {s3_output_path}")
