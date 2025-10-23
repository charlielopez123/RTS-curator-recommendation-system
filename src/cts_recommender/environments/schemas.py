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
    AUTUMN = "fall"  # Using 'fall' to match ML features and get_season() function
    WINTER = "winter"

class Channel(Enum):
    RTS1 = "RTS 1"
    RTS2 = "RTS 2"

@dataclass
class Context:
    """Programming context for decision making"""
    hour: int # 0-26 airing hour
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    season: Season
    channel: Channel

    def day_of_week_name(self) -> str:
        """Return the name of the day of the week."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[self.day_of_week]


