import string

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