# Constants specific to RTS Analytics department for parsing programming schedules, extraction of movie specific data

# Catalog ID prefix for WhatsOn movies without TMDB IDs
WHATSON_CATALOG_ID_PREFIX = "WO_"

RTS_MOVIE_CLASSKEYS = ['71', '72', '761']
RTS_WHATSON_CLASSKEYS = ['72 - Téléfilms', '71 - Films de cinéma', "761 - Longs métrages d'animation"]
RTS_COMPETITOR_MOVIE_CODES = ['AAA', 'AAB', 'AAC', 'AAD', 'AAF', 'AAG', 'AAH', 'AAL', 'ABA', 'ABB' 'ABC', 'ABD', 'ABF', 'ABM']
ALL_MOVIE_CLASSKEYS = RTS_MOVIE_CLASSKEYS + RTS_COMPETITOR_MOVIE_CODES + RTS_WHATSON_CLASSKEYS

# Specific titles where it is not the actual title of the content but a Special categorisation, in these cases retrieve from description column
RTS_SPECIAL_MOVIE_NAMES = ['Le film de minuit, 40 ans de grands frissons: Film de minuit', 'Edition spéciale Noël', 'Box office', 'Nocturne', 'Film de minuit',
       'Les classiques du cinéma', 'Ramdam, les films...',
       'Ramdam - La culture en films', 'Le film de minuit, 40 ans de grands frissons: Film de minuit']


HISTORICAL_PROGRAMMING_COLUMNS = [
        'catalog_id',
        'date',
        'start_time',
        'channel',
        'title',
        'processed_title',
        'duration_min',
        'rt_m',
        'pdm',
        'hour',
        'weekday',
        'is_weekend',
        'season',
        'public_holiday',
        'tmdb_id',
        'genres',
        'release_date',
        'vote_average',
        'popularity',
        'revenue',
        'adult',
        'original_language',
        'missing_tmdb_id',
    ]

INTEREST_CHANNELS = ['RTS 1', 'RTS 2']
