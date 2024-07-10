import string
import pandas as pd

TEAM_ABBREVIATION_MAPPING = {
    ("Arizona", "Arizona Cardinals", "Cardinals"): "ARI",
    ("Atlanta", "Atlanta Falcons", "Falcons"): "ATL",
    ("Buffalo", "Buffalo Bills", "Bills"): "BUF",
    ("Baltimore", "Baltimore Ravens", "Ravens"): "BAL",
    ("Carolina", "Carolina Panthers", "Panthers"): "CAR",
    ("Chicago", "Chicago Bears", "Bears"): "CHI",
    ("Cincinnati", "Cincinnati Bengals", "Bengals"): "CIN",
    ("Cleveland", "Cleveland Browns", "Browns"): "CLE",
    ("Dallas", "Dallas Cowboys", "Cowboys"): "DAL",
    ("Denver", "Denver Broncos", "Broncos"): "DEN",
    ("Detroit", "Detroit Lions", "Lions"): "DET",
    ("GreenBay", "Green Bay Packers", "Packers"): "GNB",
    ("Houston", "Houston Texans", "Texans"): "HOU",
    ("Indianapolis", "Indianapolis Colts", "Colts"): "IND",
    ("Jacksonville", "Jacksonville Jaguars", "Jaguars"): "JAX",
    ("KansasCity", "Kansas City Chiefs", "KCChiefs", "Kansas", "Chiefs"): "KAN",
    ("LAChargers", "Los Angeles Chargers", "LosAngeles", "Chargers"): "LAC",
    ("Oakland", "Oakland Raiders"): "OAK",
    ("LARams", "Los Angeles Rams", "Rams"): "LAR",
    ("LasVegas", "Las Vegas Raiders", "LVRaiders", "Raiders"): "LVR",
    ("Miami", "Miami Dolphins", "Dolphins"): "MIA",
    ("Minnesota", "Minnesota Vikings", "Vikings"): "MIN",
    ("NewEngland", "New England Patriots", "Patriots"): "NWE",
    ("NewOrleans", "New Orleans Saints", "Saints"): "NOR",
    ("NYGiants", "New York Giants", "Giants"): "NYG",
    ("NYJets", "New York Jets", "Jets"): "NYJ",
    ("Philadelphia", "Philadelphia Eagles", "Eagles"): "PHI",
    ("Pittsburgh", "Pittsburgh Steelers", "Steelers"): "PIT",
    ("San Diego", "San Diego Chargers", "SanDiego", "Chargers"): "SDG",
    ("SanFrancisco", "San Francisco 49ers", "49ers"): "SFO",
    ("St Louis", "St. Louis Rams", "St.Louis", "Rams"): "STL",
    ("Seattle", "Seattle Seahawks", "Seahawks"): "SEA",
    ("TampaBay", "Tampa Bay Buccaneers", "Tampa", "Buccaneers"): "TAM",
    ("Tennessee", "Tennessee Titans", "Titans"): "TEN",
    (
        "Washington",
        "Washingtom",
        "Washington Football Team",
        "Washington Redskins",
        "Washington Commanders",
        "Commanders",
    ): "WAS",
}


# Simplified function to retrieve abbreviations
def retrieve_team_abbreviation(team_name: str) -> str:
    """Retrieves the team abbreviation from the team name.

    Args:
        team_name (str): The full team name or team's home city.

    Raises:
        ValueError: If team name is not found.

    Returns:
        str: The team abbreviation.
    """
    try:
        return TEAM_ABBREVIATION_MAPPING[team_name]
    except KeyError:
        raise ValueError(f"Team name {team_name} not found in team_abbreviation_mapping")

# Updated 2-letter to 3-letter mapping
def map_abbr2_to_abbr3(team_name: str) -> str:
    """Maps the 2-letter abbreviation to the 3-letter abbreviation."""
    abbr2_mapping = {
        "KC": "KAN", "LA": "LAC", "LV": "LVR", "SF": "SFO", "TB": "TAM",
        "GB": "GNB", "NE": "NWE", "NO": "NOR", "PH": "PHI", "SD": "SDG",
    }
    return abbr2_mapping.get(team_name, team_name)

# Consolidated function to standardize team names
def standardize_team_names(team_name):
    try:
        return retrieve_team_abbreviation(team_name)
    except ValueError:
        return map_abbr2_to_abbr3(team_name)


def remove_punctuation(text: str) -> str:
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def format_player_name(name: str) -> str:
    # Split the name and take the first and last names, assuming they exist
    name_parts = name.split()
    first_name_abbr = name_parts[0][:3] if len(name_parts) > 0 else ''
    last_name_abbr = name_parts[1][:4] if len(name_parts) > 1 else ''
    # Combine and convert to upper case
    formatted_name = (first_name_abbr + last_name_abbr).upper()
    return formatted_name

def rename_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    for col in df.columns:
        if col in column_mapping:
            df = df.rename(columns={col: column_mapping[col]})
    return df