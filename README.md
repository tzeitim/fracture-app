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
- `--file_path PATH` - Path to parquet file to load automatically (optional)
- `--start_anchor SEQUENCE` - Default sequence for Start Anchor/5' end (default: GAGACTGCATGG)
- `--end_anchor SEQUENCE` - Default sequence for End Anchor/3' end (default: TTTAGTGAGGGT)
- `--umi UMI_ID` - Default UMI to select when data is loaded (optional)
- `--assembly_method METHOD` - Assembly method: compression or shortest_path (default: shortest_path)
- `--min_coverage N` - Minimum coverage threshold (default: 5)
- `--kmer_size N` - K-mer size for assembly (default: 10)
- `--auto_k` - Enable automatic k-mer size selection (flag)

**Examples:**

```bash
# Run with a specific file (skips manual upload)
python app/run.app --file_path /path/to/data/parsed_reads.parquet

# Run with custom anchors
python app/run.app --start_anchor GTGAGCAGTTTTAG --end_anchor CCCTTTAGTGAGGGT

# Run with file and specific UMI pre-selected
python app/run.app --file_path data.parquet --umi AAACGGTT

# Run with custom assembly parameters
python app/run.app --assembly_method compression --min_coverage 10 --kmer_size 15

# Run with auto k-mer size enabled
python app/run.app --auto_k

# Complete workflow setup
python app/run.app --port 8080 \
  --file_path /data/experiment1/parsed_reads.parquet \
  --start_anchor GTGAGCAGTTTTAG \
  --umi AAACGGTT \
  --min_coverage 8

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