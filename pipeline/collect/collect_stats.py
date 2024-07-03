import pandas as pd
import requests
import itertools
import os
from bs4 import BeautifulSoup
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_url(position, week, year):
    return f'https://www.footballguys.com/playerhistoricalstats?pos={position}&yr={year}&startwk={week}&stopwk={week}&profile=0'

def scrape_player_stats(url):
    response = requests.get(url)
    response.raise_for_status() 
    page_content = response.content
    soup = BeautifulSoup(page_content, 'html.parser')
    table = soup.find('table')
    headers = []
    for header in table.find_all('th'):
        headers.append(header.text.strip())
    rows = []
    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        rows.append([col.text.strip() for col in columns])
    df = pd.DataFrame(rows, columns=headers)
    return df


def collect_stats():
    logger.info('Collecting stats data')
    dp_root = Path(os.getenv('DATA_PIPELINE_ROOT'), Path.cwd().parent.parent)
    write_path = dp_root / 'data' / 'raw' / 'stats'
    write_path.mkdir(parents=True, exist_ok=True)
    positions = ['qb', 'rb', 'wr', 'te', 'pk']
    weeks = list(range(1, 18))
    seasons = list(range(2022, 2024))
    combinations = list(itertools.product(positions, weeks, seasons))
    for position, week, season in combinations:
        file_path = write_path / f'{position}_{week}_{season}.csv'
        if file_path.exists():
            logger.info(f'{position} stats for week {week} in {season} already exists')
            continue
        url = create_url(position=position, week=week, year=season)
        df = scrape_player_stats(url)
        logger.info(f"Writing stats data to {file_path}")
        df.to_csv(file_path, index=False)  # Saving to the correct file path
    logger.info('Stats data collection complete')


if __name__ == "__main__":
    collect_stats()

