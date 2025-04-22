from datetime import datetime
from typing import Any
from hytek_parser._utils import extract, select_from_enum, int_or_none
from hytek_parser.hy3.enums import Gender
from hytek_parser.hy3.schemas import ParsedHytekFile, Swimmer

def d1_parser(
    line: str, file: ParsedHytekFile, opts: dict[str, Any]
) -> ParsedHytekFile:
    """Parse a D1 swimmer entry line."""
    team_code, _ = file.meet.last_team
    swimmer = Swimmer()

    # Format in file: D1F10494Moran          Mia
    swimmer.gender = select_from_enum(Gender, extract(line, 2, 1))  # Position 2 for F/M
    swimmer.meet_id = int(extract(line, 3, 5))  # Next 5 digits after F/M
    
    swimmer.last_name = extract(line, 8, 15).strip()
    swimmer.first_name = extract(line, 23, 15).strip()
    swimmer.nick_name = ""  # Not present in your format
    swimmer.middle_initial = ""  # Not present in your format
    swimmer.usa_swimming_id = ""  # Not present in your format
    
    # Age is in position 69-70
    age_str = extract(line, 69, 2)
    swimmer.age = int(age_str) if age_str.strip() else 0
    
    # Birth date not available in this format
    swimmer.date_of_birth = datetime(2000, 1, 1).date()  # Default date
    swimmer.team_id = None
    swimmer.team_code = team_code

    file.meet.add_swimmer(swimmer)
    return file
