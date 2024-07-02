import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_adp():
    logger.info('Collecting ADP data')
    dp_root = Path(os.getenv('DATA_PIPELINE_ROOT'))
    write_path = dp_root / 'data' / 'raw' / 'adp'
    seasons = range(2015, 2025)
    for season in seasons:
        fpath = write_path / str(season)
        if fpath.exists():
            logger.info(f'ADP data for {season} already exists')
            continue
        url = f'https://www.fantasypros.com/nfl/adp/overall.php?year={season}'
        response = requests.get(url)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if table:
                headers = [header.text for header in table.find_all('th')]
                data = []
                for row in table.find_all('tr')[1:]:  # Skipping the header row
                    cols = row.find_all('td')
                    if cols:
                        cols = [ele.text.strip() for ele in cols]
                        data.append(cols)
                try:
                    df = pd.DataFrame(data, columns=headers)
                    
                    fpath.mkdir(parents=True, exist_ok=True)
                    df.to_csv(fpath / f'adp_{season}.csv', index=False)   
                    logger.info(f'ADP data for {season} collected successfully')      
                except:
                    print(f'Error processing {season}')
    logger.info('ADP data collection complete')
    return True

if __name__ == "__main__":
    collect_adp()