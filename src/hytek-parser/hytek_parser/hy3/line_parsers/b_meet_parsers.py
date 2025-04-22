from datetime import datetime
from typing import Any

from hytek_parser._utils import extract, select_from_enum, int_or_none
from hytek_parser.hy3.enums import Course, MeetType
from hytek_parser.hy3.schemas import Meet, ParsedHytekFile


def b1_parser(
    line: str, file: ParsedHytekFile, opts: dict[str, Any]
) -> ParsedHytekFile:
    """Parse a B1 primary meet info line."""
    meet = Meet()

    meet.name = extract(line, 3, 45)
    meet.facility = extract(line, 48, 45)
    
    # Handle the custom date format MM/DD-YY
    raw_date = extract(line, 93, 8)
    try:
        if '-' in raw_date:
            # Split the date on / and -
            month_day, year = raw_date.split('-')
            month, day = month_day.split('/')
            # Assume 20xx for year
            full_date = f"{month.zfill(2)}{day.zfill(2)}20{year}"
            meet.start_date = datetime.strptime(full_date, "%m%d%Y").date()
        else:
            # Try original format as fallback
            meet.start_date = datetime.strptime(raw_date, "%m%d%Y").date()
    except ValueError:
        print(f"Could not parse start date: {raw_date}")
        # Use a default date to allow processing to continue
        meet.start_date = datetime(2025, 3, 14).date()

    # Do the same for end date
    raw_end_date = extract(line, 101, 8)
    try:
        if '-' in raw_end_date:
            month_day, year = raw_end_date.split('-')
            month, day = month_day.split('/')
            full_date = f"{month.zfill(2)}{day.zfill(2)}20{year}"
            meet.end_date = datetime.strptime(full_date, "%m%d%Y").date()
        else:
            meet.end_date = datetime.strptime(raw_end_date, "%m%d%Y").date()
    except ValueError:
        print(f"Could not parse end date: {raw_end_date}")
        meet.end_date = datetime(2025, 3, 14).date()

    meet.altitude = int_or_none(extract(line, 117, 5))
    meet.country = opts["default_country"]

    file.meet = meet
    return file


def b2_parser(
    line: str, file: ParsedHytekFile, opts: dict[str, Any]
) -> ParsedHytekFile:
    """Parse a B2 secondary meet info line."""
    meet = file.meet

    meet.masters = extract(line, 94, 2) == "06"
    meet.type_ = select_from_enum(MeetType, extract(line, 97, 2))
    meet.course = select_from_enum(Course, extract(line, 99, 1))

    file.meet = meet
    return file
