import os
from pathlib import Path
import pandas as pd
from ..pipeline_utils import remove_punctuation, format_player_name
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_fantasy_draft_id(row):
    name = remove_punctuation(row['name'])
    formatted_name = format_player_name(name)
    position = str(row['position'])
    season = str(row['season'])  # Ensure season is converted to string directly
    draft_id = formatted_name + position + season
    return draft_id

def process_adp():
    logger.info('Processing ADP data')
    dp_root = Path(os.getenv('DATA_PIPELINE_ROOT', Path.cwd() / "data"))  
    read_path = dp_root  / "raw" / "adp"
    write_path = dp_root / "processed" / "adp"   
    input_raw_adp_files = [file for file in read_path.glob('**/*.csv') if file.is_file()]     
    for file in input_raw_adp_files:
        season = file.parent.name 
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'pos': 'position'})
        df['position'] = df['position'].str.replace(r'\d+', '')
        df['team'] = df['player team (bye)'].str.extract('([A-Z]{2,3})')
        df['player'] = df['player team (bye)'].str.extract(r'^(.*?)(?:\s+[A-Z]{2,3}\s*\(\d+\))?(?:\s+\w)?$')
        df = df.rename(columns={'player': 'name'})
        df['season'] = season
        df = df[['name', 'position', 'season', 'rank']]
        df.loc[df['position'] == 'PK', 'position'] = 'K'
        df = df.dropna(subset=['name'])
        df['draft_id'] = df.apply(create_fantasy_draft_id, axis=1)
        df = df.drop(columns=['name', 'position', 'season'])        
        write_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(write_path / f'adp_{season}.csv', index=False)

if __name__ == "__main__":
    process_adp()
