# ROBUST EPISODE DETECTION PATTERNS
EPISODE_PATTERNS = [
    r'ép\.\s*\d+',           # ép.1, ép. 2, etc.
    r'épisode\s*\d+',        # épisode 1, épisode 2
    r'^ep\.\s*\d+',          # ep.1, ep. 2 (at start)
    r'^E\d+',                # E1, E23, etc. (at start)
    r'S\d+E\d+',             # S01E01, S1E2, etc.
    r'^\d+$',                # Just a number (like "1", "2")
    r'^\d+\s*-\s*',          # "1 - Something", "2 - Title"
    r'partie\s*\d+',         # partie 1, partie 2
    r'\bpart\s*\d+',         # part 1, part 2
]

# Date patterns in title
DATE_PATTERNS = [
    r'\d{4}-\d{2}-\d{2}',    # YYYY-MM-DD
    r'\d{2}-\d{2}-\d{2}',    # YY-MM-DD or DD-MM-YY
    r'\d{2}\.\d{2}\.\d{2}',  # YY.MM.DD or DD.MM.YY
    r'\d{2}/\d{2}/\d{2}',    # YY/MM/DD or DD/MM/YY
]

# Known film collections (whitelist)
FILM_COLLECTIONS = [
    'Film',
    'Téléfilm', 
    'Fiction',
    'Fiction\\Achats',
    'Cinéma',
    'Film d\'action',
    'Emotions fortes',
    'Comédie',
    'Film Jeunesse',
    'Film de minuit',
    'Ecran TV',
    'Les classiques du cinéma',
    'Court métrage',
    'Nocturne',
    'Box office'
]

# Keywords that suggest a collection is film-related
FILM_COLLECTION_KEYWORDS = [
    'film', 'cinéma', 'cinema', 'fiction', 'téléfilm',
    'comédie', 'action', 'classique', 'nocturne', 'écran'
]