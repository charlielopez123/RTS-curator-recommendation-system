## Schema defintion for What's On csv catalog data

# Original column names in the What's On catalog CSV
ORIGINAL_WHATSON_COLUMNS = ['Collection', 'Titre', 'Titre Original', 'Durée', 'Description courte',
'Code contenu', 'Régions de Production', 'Année de Production',
'Contrôle Parental', 'Départment', 'Réalisation', 'Acteurs/Actrices',
'Début des droits TV', 'Fin des droits TV', 'Nb de diffusions total',
'Nb de diffusions consommées', 'Nb de diffusions disponibles',
'Nb diff. RTS1/RTS2', 'Date de 1ère diffusion', 'Date rediffusion 1',
'Date rediffusion 2', 'Date rediffusion 3', 'Date rediffusion 4',
'Date dernière diff', 'Dernière diff Rating', 'Dernière diff Rating+7',
'Nb Droits TV', 'Nb Droits TV valides', 'Référence Externe']

# Mapping from original csv column names to desired Pythonic column names
WHATSON_RENAME_MAP: dict[str, str] = {
    'Collection': 'collection',
    'Titre': 'title',
    'Titre Original': 'original_title',
    'Durée': 'duration',
    'Description courte': 'short_description',
    'Code contenu': 'class_key',
    'Régions de Production': 'production_regions',
    'Année de Production': 'production_year',
    'Contrôle Parental': 'parental_control',
    'Départment': 'department',
    'Réalisation': 'director',
    'Acteurs/Actrices': 'actors',
    'Début des droits TV': 'tv_rights_start',
    'Fin des droits TV': 'tv_rights_end',
    'Nb de diffusions total': 'total_broadcasts',
    'Nb de diffusions consommées': 'consumed_broadcasts',
    'Nb de diffusions disponibles': 'available_broadcasts',
    'Nb diff. RTS1/RTS2': 'rts1_rts2_broadcasts',
    'Date de 1ère diffusion': 'first_broadcast_date',
    'Date rediffusion 1': 'rebroadcast_date_1',
    'Date rediffusion 2': 'rebroadcast_date_2',
    'Date rediffusion 3': 'rebroadcast_date_3',
    'Date rediffusion 4': 'rebroadcast_date_4',
    'Date dernière diff': 'last_broadcast_date',
    'Dernière diff Rating': 'last_broadcast_rating',
    'Dernière diff Rating+7': 'last_broadcast_rating_plus_7',
    'Nb Droits TV': 'tv_rights_count',
    'Nb Droits TV valides': 'valid_tv_rights_count',
    'Référence Externe': 'external_reference'
}
