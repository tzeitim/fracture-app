# FRACTURE Explorer - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Loading Data](#loading-data)
5. [User Interface Overview](#user-interface-overview)
6. [Overview Tab](#overview-tab)
7. [Graph Explorer Tab](#graph-explorer-tab)
8. [Assembly Results Tab](#assembly-results-tab)
9. [Graph Controls](#graph-controls)
10. [Command-Line Arguments](#command-line-arguments)
11. [Workflows](#workflows)
12. [Tips and Best Practices](#tips-and-best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Introduction

FRACTURE Explorer is an interactive web application for visualizing and analyzing molecular assembly graphs generated using the FRACTURE protocol. The tool provides comprehensive features for:

- Loading and exploring UMI-based sequencing data
- Visualizing assembly graphs with multiple layout algorithms
- Performing contig assembly with customizable parameters
- Running parameter sweeps to optimize assembly settings
- Analyzing coverage distributions and read statistics
- Interactive node selection and path exploration

---

## Installation

### Requirements

FRACTURE Explorer requires Python 3.8 or higher and the following packages:

- shiny
- shinyswatch
- shinywidgets
- polars
- plotly
- matplotlib
- networkx
- ogtk
- pydot
- pandas
- numpy

### Installation Steps

1. Clone or download the repository
2. Install dependencies using conda/mamba:
   ```bash
   mamba env create -f mamba_env.yml
   mamba activate fracture
   ```
   
   Or using pip with pyproject.toml:
   ```bash
   pip install -e .
   ```

---

## Getting Started

### Starting the Application

To launch FRACTURE Explorer, run:

```bash
python app/run.app
```

By default, the application will start on port 8000. Open your web browser and navigate to:

```
http://localhost:8000
```

You'll see URLs for accessing the app from other devices on the same network.

### First Time Setup

1. The app will attempt to load a database from `parquet_db.txt` in the parent directory
2. If no database is found, you can load data through the UI
3. Select your data source (Local, Remote, or Upload)
4. Load your Parquet file containing UMI data

---

## Loading Data

FRACTURE Explorer supports three methods for loading data:

### 1. Remote Data

Select a system prefix from the dropdown (e.g., `/Users/pedro/data/`) and enter a relative path to your Parquet file.

**Example:**
- System Prefix: `/Users/pedro/data/`
- Remote Path: `experiment1/parsed_reads.parquet`

### 2. Local File

Browse and select a Parquet file from your local filesystem.

### 3. Upload

Drag and drop or select a Parquet file to upload directly to the application.

### Data Sampling Options

- **Sub-sample UMIs**: Enable to work with a subset of UMIs (useful for large datasets)
- **Number of UMIs to sample**: How many UMIs to randomly select
- **Minimum reads per UMI**: Only include UMIs with at least this many reads
- **Provided UMI**: Specify a particular UMI to load (optional)

### Loading the Data

1. Configure your data source and sampling options
2. Click the **"Load Dataset"** button
3. Wait for the notification confirming successful data load
4. The UMI dropdown will populate with available UMIs

---

## User Interface Overview

The application has a sidebar and three main tabs:

### Sidebar

The left sidebar contains:
- **Data Tab**: Data loading controls and settings
- **Graph Source Tab**: Graph generation and upload controls
- **Graph Controls Tab**: Visualization and layout settings
- **Parameter Sweep Tab**: Automated parameter optimization
- **Theme Controls Tab**: Visual theme selection

### Main Tabs

1. **Overview**: Summary statistics and distribution plots
2. **Graph Explorer**: Interactive graph visualization
3. **Assembly Results**: Contig assembly and analysis

---

## Overview Tab

The Overview tab provides high-level insights into your dataset:

### Summary Statistics

- **Total UMIs**: Number of unique UMIs in the loaded dataset
- **Median Reads/UMI**: Median number of reads per UMI
- **Read Length**: Length of sequencing reads

### Visualizations

#### Coverage Distribution Plot

Shows the cumulative distribution of reads across UMIs. Useful for understanding:
- How many UMIs have sufficient coverage
- Distribution of sequencing depth
- Quality of the library

#### Reads Per UMI Plot

Bar chart showing the distribution of read counts per UMI. Helps identify:
- Well-covered UMIs (high read counts)
- Outliers or problematic UMIs
- Overall coverage quality

---

## Graph Explorer Tab

The Graph Explorer is the main visualization interface for assembly graphs.

### Graph Display

The main panel shows the interactive assembly graph with:
- **Nodes**: Represent k-mer sequences
- **Edges**: Represent connections between k-mers
- **Colors**: Different colors for different sequences or features
- **Node Size**: Scaled by coverage (number of reads supporting that k-mer)

### Interactive Features

#### Node Selection

- **Click nodes** to select/deselect them
- Selected nodes are highlighted in red with larger markers
- Multiple nodes can be selected
- Selection persists across layout changes

#### Manual Node Input

Use the "Selected Nodes" text input to:
- Enter comma-separated node IDs
- View currently selected nodes
- Clear selection by emptying the field

#### Zooming and Panning

- **Scroll**: Zoom in/out
- **Click and drag**: Pan around the graph
- **Double-click**: Reset view

### Graph Height Control

Adjust the graph display height using the "Graph Height (px)" slider in the Graph Controls:
- Default: 1600 pixels
- Range: 400-3000 pixels
- Updates in real-time

---

## Assembly Results Tab

This tab provides tools for contig assembly and analysis.

### Assembly Controls

#### Assembly Method

- **Graph Compression**: Compresses the assembly graph
- **Shortest Path**: Finds the shortest path between anchors

#### Anchor Sequences

- **Start Anchor**: 5' end sequence (default: GAGACTGCATGG)
- **End Anchor**: 3' end sequence (default: TTTAGTGAGGGT)
- These can be customized via command-line or edited in the UI

#### UMI Selection

Select which UMI to assemble from the dropdown.

#### Assembly Parameters

- **Minimum Coverage**: Minimum read coverage to include k-mers
- **K-mer Size**: Length of k-mers for assembly (typically 10-50)
- **Auto K-mer Size**: Let the algorithm choose optimal k-mer size

### Running Assembly

1. Select your UMI
2. Set anchor sequences
3. Configure parameters (coverage, k-mer size)
4. Click **"Assemble Contig"**
5. Results appear in the Assembly Stats panel

### Assembly Statistics

After assembly, you'll see:
- **Assembly Status**: Success or error messages
- **Contig Length**: Length of assembled sequence
- **Path Information**: Details about the assembly path
- **Sequence**: The assembled contig sequence

### Selection UMI Stats

View detailed statistics for the selected UMI:
- Read count
- Coverage distribution
- Quality metrics

---

## Graph Controls

Located in the sidebar's "Graph Controls" tab.

### Graph Source

#### Generate from Assembly

- Uses the assembled graph from your selected UMI
- Automatically updates when you run assembly

#### Upload DOT File

- Upload a pre-generated DOT file
- Useful for exploring external graphs

### Layout Algorithm

Choose from several layout algorithms:

- **Fruchterman-Reingold** (Recommended): Balanced, good for most graphs
- **Spring Layout**: Physics-based, organic appearance
- **Kamada-Kawai**: Optimizes edge length uniformity
- **Circular**: Nodes arranged in a circle
- **Random**: Random positioning (useful as baseline)

### Layout Parameters

Fine-tune the graph layout:

- **Node Spacing (k)**: Distance between nodes (0-5.0)
  - Smaller values: nodes closer together
  - Larger values: more spread out
  
- **Layout Iterations**: Number of optimization iterations (10-2000)
  - More iterations: better layout quality but slower
  - Fewer iterations: faster but may not converge
  
- **Layout Scale**: Overall graph size (0.5-5.0)
  - Larger values: bigger graph
  
- **Component Spacing**: Space between disconnected graph components (0.5-10.0)

- **Min Component Size**: Minimum nodes in a component to display (1-20)
  - Filters out small disconnected components

### Visualization Options

- **Separate Disjoint Graphs**: Display disconnected components separately
- **Use Static Image**: Render as PNG instead of interactive widget (better for very large graphs)
- **Show Node Labels**: Display node IDs on the graph
- **Use Weighted Edges**: Adjust edge thickness based on coverage

### Edge Styling

- **Linear**: Straight lines between nodes
- **Spline**: Curved edges (smooth appearance)
- **Inverse Spline**: Alternative curve direction

### Theme Selection

Choose from multiple color themes:
- **Dark** (default): Dark background, light text
- **Latte**: Light theme
- **Mocha**: Alternative dark theme

---

## Command-Line Arguments

FRACTURE Explorer supports command-line configuration for streamlined workflows.

### Usage

```bash
python app/run.app [OPTIONS]
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | integer | 8000 | Port number for the web server |
| `--file_path` | string | None | Path to parquet file to load automatically |
| `--start_anchor` | string | GAGACTGCATGG | Default Start Anchor sequence (5' end) |
| `--end_anchor` | string | TTTAGTGAGGGT | Default End Anchor sequence (3' end) |
| `--umi` | string | None | Default UMI to select when data loads |
| `--assembly_method` | string | shortest_path | Assembly method: compression or shortest_path |
| `--min_coverage` | integer | 5 | Minimum coverage threshold for assembly |
| `--kmer_size` | integer | 10 | K-mer size for assembly |
| `--auto_k` | flag | False | Enable automatic k-mer size selection |

### Examples

**Basic usage with defaults:**
```bash
python app/run.app
```

**Custom port:**
```bash
python app/run.app --port 8080
```

**Load a specific file (skips manual upload):**
```bash
python app/run.app --file_path /path/to/data/parsed_reads.parquet
```

**Custom anchor sequences:**
```bash
python app/run.app --start_anchor GTGAGCAGTTTTAG --end_anchor CCCTTTAGTGAGGGT
```

**Pre-select a specific UMI:**
```bash
python app/run.app --umi AAACGGTTCCAA
```

**Custom assembly parameters:**
```bash
python app/run.app --assembly_method compression --min_coverage 10 --kmer_size 15
```

**Enable auto k-mer size:**
```bash
python app/run.app --auto_k
```

**All options combined:**
```bash
python app/run.app --port 8080 \
  --file_path /data/experiment1/parsed_reads.parquet \
  --start_anchor GTGAGCAGTTTTAG \
  --end_anchor CCCTTTAGTGAGGGT \
  --umi AAACGGTTCCAA \
  --assembly_method compression \
  --min_coverage 8 \
  --kmer_size 12 \
  --auto_k
```

**View help:**
```bash
python app/run.app --help
```

### Notes

- Command-line arguments set **defaults** in the UI, but users can still change them
- The `--file_path` option populates the "Local Parquet File" input box and selects "Local File Path" as the input method
- If a specified UMI doesn't exist in the data, the first available UMI will be selected
- The `--auto_k` flag overrides the `--kmer_size` parameter when enabled
- The app logs which values are being used at startup
- You still need to click "Load Dataset" button after the app starts to actually load the file

---
---

## Workflows

### Basic Assembly Workflow

1. **Load Data**
   - Select data source and load Parquet file
   - Choose a UMI from the dropdown

2. **Configure Assembly**
   - Set Start and End Anchor sequences
   - Adjust k-mer size and coverage threshold
   - Select assembly method

3. **Run Assembly**
   - Click "Assemble Contig"
   - Review assembly statistics

4. **Visualize Graph**
   - Go to Graph Explorer tab
   - Adjust layout if needed
   - Explore nodes and paths

### Parameter Optimization Workflow

1. **Set Parameter Ranges**
   - Go to Parameter Sweep tab
   - Define k-mer range (start, end, step)
   - Set coverage range

2. **Run Sweep**
   - Click "Run Parameter Sweep"
   - Wait for completion (may take time for large ranges)

3. **Analyze Results**
   - View the heatmap showing assembly success
   - Identify optimal parameter combinations
   - Use these values for final assembly

### Interactive Exploration Workflow

1. **Load and Assemble**
   - Load data and run initial assembly

2. **Visualize Graph**
   - Open Graph Explorer
   - Try different layout algorithms
   - Adjust node spacing and iterations

3. **Select Nodes**
   - Click nodes of interest
   - View selected node information
   - Use selections to guide further analysis

4. **Refine Assembly**
   - Return to Assembly Results
   - Adjust parameters based on graph insights
   - Re-run assembly

### Coverage Analysis Workflow

1. **Load Data with Sampling**
   - Enable UMI sub-sampling
   - Set appropriate sample size

2. **Review Overview**
   - Check coverage distribution
   - Identify well-covered UMIs

3. **Enable Coverage Plot**
   - In Data tab, enable "Enable Coverage Plot"
   - Visualize coverage across positions

4. **Filter by Coverage**
   - Adjust "Minimum Coverage" in assembly controls
   - Re-run assembly with filtered data

---

## Tips and Best Practices

### Performance Optimization

- **Use UMI sampling** for initial exploration of large datasets
- **Start with smaller k-mer sizes** (10-15) for faster assembly
- **Enable static images** for very large graphs (>1000 nodes)
- **Reduce layout iterations** for quick previews, increase for final visualization

### Assembly Quality

- **Choose appropriate k-mer size**:
  - Too small: May create spurious connections
  - Too large: May fragment the assembly
  - Use Auto K-mer Size for automatic selection

- **Set minimum coverage** based on your data:
  - Higher coverage: More stringent, fewer errors
  - Lower coverage: More complete but may include noise

- **Validate anchors**: Ensure Start and End Anchors actually exist in your data

### Graph Visualization

- **Try different layouts**: Each algorithm works better for different graph structures
- **Adjust node spacing**: Start with default (0.001), increase for clarity
- **Use component separation**: Helpful for graphs with multiple disconnected regions
- **Increase graph height**: For complex graphs, use 1600-2400 pixels

### Troubleshooting Assembly

If assembly fails or produces unexpected results:

1. Check that anchor sequences are correct
2. Verify UMI has sufficient coverage
3. Try different k-mer sizes
4. Reduce minimum coverage threshold
5. Examine the graph visually for issues
6. Check the logs for error messages

---

## Troubleshooting

### Common Issues and Solutions

#### "No data loaded" or empty UMI dropdown

**Solution:**
- Verify the Parquet file path is correct
- Ensure the file contains required columns (`umi`, `read`, `sequence`)
- Check file permissions
- Review error messages in the browser console or server logs

#### Graph not displaying

**Solution:**
- Wait for assembly to complete (check notification)
- Try switching between interactive and static image modes
- Reduce graph complexity by increasing minimum coverage
- Check browser console for JavaScript errors

#### Assembly takes too long

**Solution:**
- Reduce k-mer size
- Increase minimum coverage (filters out low-quality k-mers)
- Use a smaller UMI sample
- Try graph compression method instead of shortest path

#### Selected UMI from command-line not working

**Solution:**
- Verify the UMI exists in the loaded dataset
- Check spelling and case sensitivity
- Review logs for warnings about UMI selection
- The app will fall back to first UMI if specified one isn't found

#### Graph layout looks messy

**Solution:**
- Increase layout iterations (try 500-1000)
- Adjust node spacing parameter
- Try different layout algorithms
- Enable "Separate Disjoint Graphs" for disconnected components
- Increase graph height for better visibility

#### Port already in use

**Solution:**
- Use `--port` to specify a different port
- Kill the process using the port: `lsof -ti:8000 | xargs kill`
- Check if another instance is already running

#### Browser disconnects or times out

**Solution:**
- The app includes keep-alive settings (5 minutes)
- Large operations may take time - be patient
- Check network connectivity
- Try refreshing the browser

### Getting Help

For additional support:

1. Check the application logs in the terminal
2. Review error messages in browser console (F12)
3. Verify all dependencies are installed correctly
4. Ensure data files are in the expected format
5. Try with a smaller test dataset first

### Reporting Issues

When reporting problems, please include:
- Command used to start the app
- Browser type and version
- Error messages from terminal and browser
- Steps to reproduce the issue
- Sample data if possible (or description of data format)

---

## Appendix: Keyboard Shortcuts

While the app primarily uses mouse interaction, some browser shortcuts are useful:

- **Ctrl/Cmd + Plus**: Zoom in on page
- **Ctrl/Cmd + Minus**: Zoom out on page
- **Ctrl/Cmd + 0**: Reset page zoom
- **F5**: Refresh page
- **F12**: Open browser developer tools (useful for debugging)

---

## Appendix: Data Format

FRACTURE Explorer expects Parquet files with the following columns:

- `umi`: Unique Molecular Identifier (string)
- `read`: Read identifier (string)
- `sequence`: Nucleotide sequence (string)
- Additional optional columns for metadata

Example data structure:
```
umi          | read      | sequence
-------------|-----------|------------------
AAACGGTT     | read_001  | ATCGATCGATCG...
AAACGGTT     | read_002  | ATCGATCGATCG...
TTGGCCAA     | read_003  | GCTAGCTAGCTA...
```

---

**End of User Manual**

For quick reference, see [README.md](README.md).

For technical documentation, see the module docstrings in the source code.
