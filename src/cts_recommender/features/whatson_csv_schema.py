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

WHATSON_RENAME_MAP: dict[str, str] = {
    'Collection',
    'Titre',
    'Titre Original',
    'Durée',
    'Description courte',
    'Code contenu',
    'Régions de Production',
    'Année de Production',
    'Contrôle Parental',
    'Départment',
    'Réalisation',
    'Acteurs/Actrices',
    'Début des droits TV',
    'Fin des droits TV',
    'Nb de diffusions total',
    'Nb de diffusions consommées',
    'Nb de diffusions disponibles',
    'Nb diff. RTS1/RTS2',
    'Date de 1ère diffusion',
    'Date rediffusion 1',
    'Date rediffusion 2',
    'Date rediffusion 3',
    'Date rediffusion 4',
    'Date dernière diff',
    'Dernière diff Rating',
    'Dernière diff Rating+7',
    'Nb Droits TV',
    'Nb Droits TV valides',
    'Référence Externe'
}