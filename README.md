# FRACTURE Explorer

Shiny app for interactive assembly of molecules using the fracture protocol.

## Overview

FRACTURE Explorer is an interactive tool for visualizing and analyzing the assembly of molecules using the fracture protocol. The application provides a user-friendly interface for exploring assembly graphs, running parameter sweeps, and analyzing contig results.

## Features

- Data loading from local, remote, or uploaded files
- Interactive assembly graph visualization
- Parameter sweep for optimization of assembly parameters
- Coverage analysis and visualization
- Contig exploration with sequence highlighting

## Project Structure

The application is organized into modules for better maintainability:

```
app/
├── modules/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration constants
│   ├── data_loader.py      # Data loading functions
│   ├── data_processing.py  # Data processing functions
│   ├── ui_components.py    # UI component definitions
│   └── visualization.py    # Visualization functions
├── app.py                  # Main application file
├── run.app                 # Executable run script
└── ...                     # Data files, etc.
```

## Modules

- **config.py**: Contains configuration constants like system paths, color schemes, and other settings.
- **data_loader.py**: Functions for loading data from different sources.
- **data_processing.py**: Core data processing and analysis functions.
- **ui_components.py**: UI component definitions to build a consistent interface.
- **visualization.py**: Functions for creating interactive visualizations.

## Running the App

To run the application, execute:

```bash
python app/run.app
```

The app will be available at http://localhost:8000 by default.

### Command-Line Arguments

The application supports several optional command-line arguments:

```bash
python app/run.app [OPTIONS]
```

**Options:**

- `--port PORT` - Port to run the server on (default: 8000)
- `--start_anchor SEQUENCE` - Default sequence for Start Anchor/5' end (default: GAGACTGCATGG)
- `--end_anchor SEQUENCE` - Default sequence for End Anchor/3' end (default: TTTAGTGAGGGT)
- `--umi UMI_ID` - Default UMI to select when data is loaded (optional)

**Examples:**

```bash
# Run with custom anchors
python app/run.app --start_anchor GTGAGCAGTTTTAG --end_anchor CCCTTTAGTGAGGGT

# Run with a specific UMI pre-selected
python app/run.app --umi AAACGGTT

# Run on a different port with custom settings
python app/run.app --port 8080 --start_anchor GTGAGCAGTTTTAG --umi AAACGGTT

# View all options
python app/run.app --help
```

For detailed usage instructions, see [USER_MANUAL.md](USER_MANUAL.md).

## Requirements

The application requires the following Python packages:

- shiny
- shinyswatch
- shinywidgets
- polars
- plotly
- matplotlib
- networkx
- ogtk
- pydot

See `pyproject.toml` for the complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.