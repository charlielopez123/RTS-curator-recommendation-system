import re
import unicodedata

def tmdb_clean_up_title(title):
    """
    Clean up movie titles by removing parenthetical information.
    
    This function removes content in parentheses that contains at least 3 letters,
    which typically includes things like "(2021)", "(Director's Cut)", "(Extended Edition)", etc.
    
    Args:
        title (str): The original movie title
        
    Returns:
        str: Cleaned title with parenthetical content removed
    """
    # Regex pattern to match parentheses containing at least 3 letters
    # (?=(?:.*[A-Za-z]){3,}) is a positive lookahead ensuring at least 3 letters inside
    pattern = r"\((?=(?:.*[A-Za-z]){3,}).*?\)"
    
    # Remove the matched parenthetical content, then clean up any trailing spaces before closing parens
    # Example: "Movie Title (2021) " -> "Movie Title"
    title = re.sub(r'\s+\)', '', re.sub(pattern, '', title)).strip()
    
    return title

def normalize(s: str) -> str:
    """
    Normalize a string for better comparison by removing accents and standardizing case.
    
    This function:
    1. Decomposes Unicode characters (é becomes e + ´)
    2. Removes accent marks and diacritics  
    3. Converts to lowercase using casefold() for proper Unicode handling
    
    Args:
        s (str): Input string to normalize
        
    Returns:
        str: Normalized string without accents in lowercase
        
    Example:
        "Amélie" -> "amelie"
        "CAFÉ" -> "cafe"
    """
    # Decompose Unicode characters into base + combining characters
    # NFKD = Normalization Form Canonical Decomposition
    decomposed = unicodedata.normalize("NFKD", s)
    
    # Filter out combining characters (accents, diacritics)
    # unicodedata.combining() returns non-zero for combining characters
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    
    # Use casefold() instead of lower() for better Unicode case handling
    # casefold() handles special cases like German ß -> ss
    return no_accents.casefold()