from pathlib import Path
from hytek_parser import hy3_parser
from typing import Optional

class TrackMeetAnalyzer:
    """Analyzes track meet data from HY3 files."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "hy3"
        
    def parse_meet_file(self, filename: str):
        """Parse a meet file and return the parsed data."""
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Meet file not found: {file_path}")
            
        try:
            meet = hy3_parser.parse_hy3(file_path)
            return meet
        except Exception as e:
            print(f"Error parsing {filename}: {str(e)}")
            raise

def main():
    analyzer = TrackMeetAnalyzer()
    meet = analyzer.parse_meet_file("Sharyland25COMPLETED.hy3")
    
    # Access meet data
    print("\nTeams:")
    for code, team in meet.teams.items():
        print(f"{team.name} ({code})")
        
    print("\nEvents:")
    for number, event in meet.events.items():
        print(f"Event {number}: {event.distance}m {event.stroke.name}")

if __name__ == "__main__":
    main()
