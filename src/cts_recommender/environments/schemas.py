from dataclasses import dataclass
from enum import Enum

class TimeSlot(Enum):
    PRIME_TIME = "prime_time"  # 20:00-22:00
    LATE_NIGHT = "late_night"  # 22:00-00:00
    AFTERNOON = "afternoon"    # 14:00-18:00
    MORNING = "morning"        # 06:00-12:00

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

@dataclass
class Context:
    """Programming context for decision making"""
    hour: int # 0-26 airing hour
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    season: Season


