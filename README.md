<div align="center" id="top">
  <h1>Track Analysis</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge" alt="Python Version" />
  <img src="https://img.shields.io/badge/Status-In_Development-yellow.svg?style=for-the-badge" alt="Status" />
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License" />
  </a>
</div>

## :page_with_curl: Table of Contents

- [:page_with_curl: Table of Contents](#page_with_curl-table-of-contents)
- [:dart: About](#dart-about)
- [:sparkles: Features](#sparkles-features)
- [:file_folder: Project Structure](#file_folder-project-structure)
- [:rocket: Technologies](#rocket-technologies)
- [:white_check_mark: Requirements](#white_check_mark-requirements)
- [:checkered_flag: Getting Started](#checkered_flag-getting-started)
- [:memo: License](#memo-license)

<br>

## :dart: About

This project provides tools for analyzing track and field meet data exported from Hytek's Meet Manager software. It has been specifically adapted from swimming meet analysis to work with track and field events and metrics.

The core functionality parses `.hy3` files (Hytek Meet Manager merge exports) to extract comprehensive meet data, including athletes, events, and results. This data can then be transformed and analyzed for performance tracking, statistics, and reporting.

## :sparkles: Features

:heavy_check_mark: Parse `.hy3` files from Hytek Meet Manager\
:heavy_check_mark: Extract and structure track meet data\
:heavy_check_mark: Transform raw data for performance analysis\
:heavy_check_mark: Support for both individual and relay events\
:heavy_check_mark: Track-specific metrics and standardization

## :file_folder: Project Structure

The repository is organized as follows:

- `/src/hytek-parser/` - Modified fork of the hytek-parser library adapted for track events
- `/data/` - Directory for storing meet data files (`.hy3`, `.csv`)
- `/Old/` - Legacy code that worked with CSV exports (retained for reference)

## :rocket: Technologies

The following tools and libraries are used in this project:

- [Python](https://python.org) 3.9+
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [hytek-parser](https://github.com/SwimComm/hytek-parser) (modified) - Core parsing functionality
- [Attrs](https://www.attrs.org/en/stable/)
- [AEnum](https://pypi.org/project/aenum/)

## :white_check_mark: Requirements

To use this project, you'll need:

- [Python](https://python.org) 3.9 or higher
- `.hy3` files exported from Hytek Meet Manager 6.0 or newer

## :checkered_flag: Getting Started

1. Clone this repository:

```sh
git clone https://github.com/your-username/track-analysis.git
cd track-analysis
```

2. Set up a virtual environment:

```sh
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using Poetry
poetry install
```

3. Install dependencies:

```sh
pip install -r requirements.txt
```

4. Place your `.hy3` files in the `/data/` directory

5. Run the parser:

```sh
python src/main.py
```

## :memo: License

This project is under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Based on the <a href="https://github.com/SwimComm/hytek-parser">hytek-parser</a> library by Nino Maruszewski</p>
</div>

&#xa0;

<a href="#top">Back to top</a>
