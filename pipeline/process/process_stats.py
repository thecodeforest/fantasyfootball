import os
import pandas as pd
from pathlib import Path
import logging

from pipeline.pipeline_utils import rename_columns, remove_punctuation, format_player_name
from pipeline.process.process_adp import create_fantasy_draft_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLAYER_STATS_COLUMN_MAPPING = {"rsh": "rush_attempts",
                    "rshyd": "rush_yards",
                    "exp": "years_in_league",
                    "rshtd": "rush_touchdowns",
                    "rec": "receptions",
                    "recyd": "receiving_yards",
                    "rectd": "receiving_touchdowns",
                    "fgm": "field_goals_made",
                    "fga": "field_goals_attempted",
                    "xpm": "extra_points_made",
                    "xpa": "extra_points_attempted",
                    "cmp": "completions",
                    "att": "pass_attempts",
                    "pyd": "pass_yards",
                    "ptd": "pass_touchdowns",
                    "int": "interceptions",
                    }

def create_stats_id(row):
    name = remove_punctuation(row['name'])  # Strip punctuation from the name
    formatted_name = format_player_name(name)
    year_of_birth = int(row['season']) - int(row['age'])
    player_id = formatted_name + str(year_of_birth)
    return player_id

def process_stats():
    logger.info('Processing stats data')
    dp_root = Path(os.getenv('DATA_PIPELINE_ROOT', Path.cwd()))
    logger.info(f"dp root is {dp_root}")
    # /home/runner/work/data/raw/stats
    read_path = dp_root.parent.parent / "data" / "raw" / "stats"
    logger.info(f"read path is {read_path}")
    write_path = dp_root.parent.parent / "data" / "processed" / "stats"   
    input_raw_stats_files = [file for file in read_path.glob('**/*.csv') if file.is_file()]  
    for file_path in input_raw_stats_files:
        position, year, week, fname = str(file_path).split("/")[-4:]
        df = pd.read_csv(file_path)
        df = df.rename(columns=str.lower)
        df = df.drop(columns=["rank", "fp/g", "fantpt", "fg%", "y/att", 'cm%', 'y/rsh', 'y/rec', 'g'], errors='ignore')
        df = rename_columns(df, PLAYER_STATS_COLUMN_MAPPING)
        df = df.fillna(0)
        df['position'] = position
        df['position'] = df['position'].str.upper()
        df['week'] = week
        df['season'] = year   
        df.loc[df['position'] == 'PK', 'position'] = 'K'
        df['team']  = df['name'].apply(lambda x: x.split(" ")[-1])
        df['name'] = df.apply(lambda row: row['name'].replace(row['team'], ''), axis=1)
        df['name'] = df['name'].str.strip()
        df['stats_id'] = df.apply(create_stats_id, axis=1)
        df['draft_id'] = df.apply(create_fantasy_draft_id, axis=1)
        Path(write_path / position).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(write_path) / position / fname, index=False)
    logger.info('Stats data processing complete')

if __name__ == "__main__":
    process_stats()
