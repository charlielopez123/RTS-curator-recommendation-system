from dataclasses import dataclass
from typing import Mapping, Sequence

# Original RData programming table column order (as delivered)
ORIGINAL_PROGRAMMING_COLUMNS: tuple[str, ...] = (
    "Date",
    "Title",
    "Description",
    "Channel",
    "Start time",
    "End time",
    "Duration",
    "Net Duration",
    "BrdCstClassKey",
    "Cibles",
    "Activité",
    "Rt-M",
    "PDM-%",
)

# Mapping from original RData column names to desired Pythonic column names
PROGRAMMING_RENAME_MAP: dict[str, str] = {
    "Date": "date",
    "Title": "title",
    "Description": "description",
    "Channel": "channel",
    "Start time": "start_time",
    "End time": "end_time",
    "Duration": "duration",
    "Net Duration": "net_duration",
    "BrdCstClassKey": "class_key",
    "Cibles": "target_audience",
    "Activité": "activity",
    "Rt-M": "rt_m",
    "PDM-%": "pdm",
}

