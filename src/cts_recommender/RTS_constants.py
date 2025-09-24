# Constants specific to RTS Analytics department for parsing programming schedules, extraction of movie specific data

RTS_MOVIE_CLASSKEYS = ['71', '72', '761']
RTS_COMPETITOR_MOVIE_CODES = ['AAA', 'AAB', 'AAC', 'AAD', 'AAF', 'AAG', 'AAH', 'AAL', 'ABA', 'ABB' 'ABC', 'ABD', 'ABF', 'ABM']
ALL_MOVIE_CLASSKEYS = RTS_MOVIE_CLASSKEYS + RTS_COMPETITOR_MOVIE_CODES

# Specific titles where it is not the actual title of the content but a Special categorisation, in these cases retrieve from description column
RTS_SPECIAL_MOVIE_NAMES = ['Le film de minuit, 40 ans de grands frissons: Film de minuit', 'Edition spéciale Noël', 'Box office', 'Nocturne', 'Film de minuit',
       'Les classiques du cinéma', 'Ramdam, les films...',
       'Ramdam - La culture en films', 'Le film de minuit, 40 ans de grands frissons: Film de minuit']
