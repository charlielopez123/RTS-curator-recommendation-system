import requests
import zoneinfo
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import tqdm.auto as tqdm
import re

from cts_recommender.competition import M6, TF1
from cts_recommender.competition.comp_constants import SITES

import logging

logger = logging.getLogger(__name__)

def _make_session_for(channel: str):
    sess = requests.Session()
    site_cfg = SITES[channel]
    if site_cfg['default_headers'] is not None:
        sess.headers.update(site_cfg['default_headers'])
    return sess

class CompetitorDataScraper:
    """For online training - scrape competitor programming schedules"""
    def __init__(self):
        # Create sessions for each competing channel
        self.sessions = {channel: _make_session_for(channel) for channel in SITES.keys()}
        self.scraped_schedules = {}
        self.scraped_ids = []

    def scrape_upcoming_schedules(self, days_ahead: int = 21):
        """Fetch exactly one XML per TV-week per channel, extract films."""

        # Collect all TV‑week starts (Saturdays) 
        # Weeks defined as Saturday to Friday
        tv_week_starts = self._available_tv_week_starts(days_ahead=days_ahead)
        for channel in self.sessions.keys():
            self.scraped_schedules[channel] = {}
            for week_sat in tqdm(sorted(tv_week_starts), desc= f'Weeks for {channel}', leave = True):
                week_mon = week_sat + timedelta(days=2)
                iso_year, iso_week, _ = week_mon.isocalendar()
                params = {'week_sat': week_sat, 'iso_year': iso_year, 'iso_week': iso_week}
                try:
                    # Check if already scraped
                    if f'week_{iso_week}_{channel}_{iso_year}' in self.scraped_ids:
                        continue
                    xml_path = self.download_xml(channel, **params)
                except Exception as e:
                    print(f"Failed to scrape {channel}: {e}")
                self.scraped_ids.append(f'week_{iso_week}_{channel}_{iso_year}')

    def available_tv_week_starts(self, days_ahead: int = 21, now: datetime = None) -> List[date]:
        """
        Determine all Saturdays for which the TV week is available to scrape.
        Competition schedules are available 3 weeks in advance on Saturdays at 18:00 (6pm) local time.
        """
        tz = zoneinfo.ZoneInfo("Europe/Zurich") # RTS timezone
        now = now or datetime.now(tz)
        now = now.astimezone(tz)

        start_date = now.date()
        end_date = start_date + timedelta(days=days_ahead)

        # Find first following Saturday >= start_date (Saturday is weekday 5)
        days_until_sat = (5 - start_date.weekday()) % 7
        first_sat = start_date + timedelta(days=days_until_sat)

        available_saturdays: List[date] = []
        current_sat = first_sat
        while current_sat <= end_date:
            # Availability for week starting at current_sat is (current_sat - 3 weeks) Saturday at 18:00
            availability_base = current_sat - timedelta(weeks=3)
            availability_dt = datetime(
                availability_base.year,
                availability_base.month,
                availability_base.day,
                18, 0, 0,
                tzinfo=tz
            )
            if now >= availability_dt:
                available_saturdays.append(current_sat)
            current_sat += timedelta(weeks=1)

        return available_saturdays

    def download_xml(self, channel: str, **params) -> Path | None:
        session = self.sessions[channel]
        if channel == 'TF1':
            xml_path = TF1.TF1_download_xml(session, **params)
        elif channel == 'M6':
            xml_path = M6.M6_download_xml(session, **params)
        else:
            logger.warning(f'Invalid Channel Name for xml download: {channel}')
            xml_path = None

        return xml_path

    def parse_cinema(self, channel, xml_path) -> List[Dict[str, Optional[str]]]:
        if channel == 'TF1':
            week_schedule = list(TF1.parse_cinema_from_tf1(xml_path))
        elif channel == 'M6':
            week_schedule = list(M6.parse_cinema_from_m6(xml_path))
        else:
            logger.error(f'Invalid Channel Name for parsing: {channel, xml_path}')

        return week_schedule

    def _extract_iso_week_from_filename(self, filename: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Returns (iso_year, iso_week) extracted from the filename.

        Handles:
        - Week-style: e.g., "M6_2025_31.xml" -> (2025, 31)
        """
        p = Path(filename)
        stem = p.stem  # filename without extension

        # Try week pattern at end: YYYY_WW or YYYY-WW
        m_week = re.search(r"(\d{4})[_-](\d{1,2})$", stem)
        
        if m_week:
            year = int(m_week.group(1))
            week = int(m_week.group(2))
            
            return year, week  # assume this is already ISO week

    def _extract_channel(self, filename: str) -> str:
            stem = Path(filename).stem
            if "_" in stem:
                return stem.split("_", 1)[0]
            if "-" in stem:
                return stem.split("-", 1)[0]
            return stem

    def process_competition_folder(self, folder_path: Path):
        """
        Process all XML files in the given folder, extracting cinema schedules.
        """
        for xml_path in folder_path.glob("*.xml"):
            channel = self._extract_channel(xml_path.name)
            schedule = self.parse_cinema(channel, xml_path)
            iso_year, iso_week = self._extract_iso_week_from_filename(xml_path)
            
            self.scraped_schedules[channel][f'{iso_week}'] = schedule