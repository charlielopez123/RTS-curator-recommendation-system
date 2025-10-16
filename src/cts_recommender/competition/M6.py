from requests import Session
from pathlib import Path
from lxml import etree
from typing import Dict, Optional, Generator

from cts_recommender.competition.comp_constants import SITES

def M6_download_xml(session: Session, **params) -> Path:
    str_year = str(params['iso_year'])
    str_week = str(params['iso_week'])

    base_xml = SITES['M6']['base_xml'].format(year = str_year, week = str_week)
    resp = session.get(base_xml, timeout = 10)
    resp.raise_for_status()
    out_dir = Path(f'competition_data')
    
    out_path = out_dir / f'M6_{str_year}_{str_week}.xml'
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    
    return out_path # Return path to xml file for parsing

def parse_cinema_from_m6(xml_path: Path) -> Generator[Dict[str, Optional[str]], None, None]:
    """
    Stream-parse an M6 weekly XML feed and yield all 'Cinéma' diffusions
    with their broadcast date, time, and title.
    """
    # iterparse both start/end so we can pick up the jour date and each diffusion
    context = etree.iterparse(
        xml_path,
        events=('start', 'end'),
        recover=True,
        remove_blank_text=True
    )
    for event, elem in context:
        # When we finish a <diffusion>, see if it's a film and extract
        if event == 'end' and elem.tag == 'diffusion':
            fmt = elem.findtext('format')  # e.g. "Cinéma"
            if fmt == 'Long Métrage':
                # full timestamp: "YYYY‑MM‑DD HH:MM"
                dt = elem.findtext('dateheure')
                # split into date/time (we already have current_date if you prefer)
                date_part, time_part = dt.split(' ')
                title = elem.findtext('titreprogramme')
                
                yield {
                    'date':  date_part,      # or use current_date
                    'time':  time_part,
                    'title': title
                }
            # 4) Clear memory for this <diffusion>
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        # 5) Once we finish the entire <jour>, clear it too
        elif event == 'end' and elem.tag == 'jour':
            elem.clear()