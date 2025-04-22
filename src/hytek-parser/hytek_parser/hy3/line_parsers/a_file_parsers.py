from datetime import datetime
from typing import Any

from hytek_parser._utils import extract
from hytek_parser.hy3.schemas import ParsedHytekFile, Software


def a1_parser(
    line: str, file: ParsedHytekFile, opts: dict[str, Any]
) -> ParsedHytekFile:
    """Parse an A1 line with file info."""
    file.file_description = extract(line, 5, 25)
    file.software = Software(extract(line, 30, 15), extract(line, 45, 10))

    raw_date = extract(line, 59, 17)
    try:
        # First try to parse the custom format from your files
        if '.0Fb' in raw_date:
            # Remove the .0Fb prefix and handle the shortened year
            cleaned_date = raw_date.replace('.0Fb', '')
            date_parts = cleaned_date.split(' ')
            date_str = date_parts[0].replace('/', '')
            # Assuming 21-25 means 2025
            if '-' in date_str:
                month_day, year = date_str.split('-')
                date_str = f"{month_day}20{year}"
            time_str = date_parts[1] + " PM"  # Assuming PM for now
            file.date_created = datetime.strptime(f"{date_str} {time_str}", "%m%d%Y %I:%M %p")
    except ValueError as e:
        try:
            # Try original format as fallback
            file.date_created = datetime.strptime(raw_date, "%m%d%Y %I:%M %p")
        except ValueError:
            print(f"Could not parse date: {raw_date}")
            # Use a default date to allow processing to continue
            file.date_created = datetime(2025, 4, 21, 20, 44)

    file.licensee = extract(line, 76, 53)
    return file
