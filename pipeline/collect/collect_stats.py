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
    positions = ['qb', 'rb', 'wr', 'te', 'pk']
    weeks = list(range(1, 18))
    years = list(range(2022, 2024))
    combinations = list(itertools.product(positions, weeks, years))
    for position, week, year in combinations:
        dir_path = dp_root / 'data' / 'raw' / 'stats' / str(position) / str(year) / str(week)
        file_path = dir_path / f'{position}_{week}_{year}.csv'
        if file_path.exists():
            logger.info(f'{position} stats for week {week} in {year} already exists')
            continue
        url = create_url(position=position, week=week, year=year)
        df = scrape_player_stats(url)
        dir_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)  # Saving to the correct file path
    logger.info('Stats data collection complete')


if __name__ == "__main__":
    collect_stats()

