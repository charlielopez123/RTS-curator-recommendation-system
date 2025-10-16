from requests import Session
from pathlib import Path
from lxml import etree
from typing import Generator, List, Dict, Optional

from cts_recommender.competition.comp_constants import SITES

def TF1_download_xml(session: Session, **params) -> Path:
    """
    Download the TF1 weekly XML feed for the given week parameters.
    """
    
    BASE_HTML = SITES['TF1']['base_html']
    BASE_XML = SITES['TF1']['base_xml']

    html_url = BASE_HTML.format(date = str(params['week_sat']))
    xml_url  = BASE_XML.format(date = str(params['week_sat']))
    
    out_dir = Path(f'competition_data')
    out_dir.mkdir(parents=True, exist_ok=True)

    session.get(html_url, timeout=10).raise_for_status() # sets TF1 cookies
    resp = session.get(xml_url, headers={'Accept': 'application/xml'}, timeout=10)
    resp.raise_for_status()

    str_year = str(params['iso_year'])
    str_week = str(params['iso_week'])

    out_path = out_dir / f'TF1_{str_year}_{str_week}.xml'
    with open(out_path, 'wb') as f:
        f.write(resp.content) 

    return out_path # Return path to xml file for parsing

def parse_cinema_from_tf1(xml_path) -> Generator[Dict[str, Optional[str]], None, None]:
    """
    Stream-parse the TF1 weekly XML feed and yield all 'Cinéma' emissions
    with their broadcast date, time, and title.
    """
    context = etree.iterparse(
        xml_path,
        events=('start', 'end'),
        recover=True,
        remove_blank_text=True
    )
    current_date = None
    for event, elem in context:
        # When we hit the start of a new day, grab its date
        if event == 'start' and elem.tag == 'JOUR':
            current_date = elem.get('date')  # e.g. "09/08/2025"
        # When we finish an emission, check & extract
        elif event == 'end' and elem.tag == 'EMISSION':
            # 1) Filter for 'Cinéma'
            type_em = elem.findtext('typeEmission')
            if type_em == 'Cinéma':
                time = elem.get('heureDiffusion')        # e.g. "21.10"
                title = elem.findtext('titre')           # e.g. "Thor : Ragnarök"
                yield {
                    'date':  current_date,
                    'time':  time,
                    'title': title
                }
            # 2) Clear this <EMISSION> to free memory
            elem.clear()
            # remove any preceding siblings
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        # Once we finish the entire <JOUR>, clear it too
        elif event == 'end' and elem.tag == 'JOUR':
            elem.clear()