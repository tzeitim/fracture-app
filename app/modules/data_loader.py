# Data loading functions
import polars as pl
from pathlib import Path
from shiny import reactive, ui
from .config import SYSTEM_PREFIXES

def load_database(database_path):
    """Load and prepare the database of available datasets"""
    db = (
        pl.read_csv(database_path, has_header=False)
        .rename({'column_1':'full_path'})
        .with_columns(pl.col('full_path').str.split("/").alias('split'))
        .with_columns(name=pl.col('split').list.get(-3) +"::"+ pl.col('split').list.get(-2))
        .drop('split')
        .sort('name')
    )
    
    return {i:ii for i,ii in db.iter_rows()}

def load_data(input_type, file_info, system_prefix=None, remote_path=None, 
              sample_umis=False, sample_n_umis=100, sample_min_reads=100):
    """Load data based on input type and parameters"""
    try:
        if input_type == "upload" and file_info is not None:
            file_path = file_info[0]["datapath"]
        elif input_type == "local" and file_info is not None:
            file_path = file_info
        elif input_type == "remote" and remote_path:
            prefix = SYSTEM_PREFIXES[system_prefix]
            file_path = prefix + remote_path
        else:
            return None, "No valid input choices"

        if not Path(file_path).exists():
            return None, f"File not found: {file_path}"

        if sample_umis:
            df = (
                pl.scan_parquet(file_path)
                .filter(pl.col('reads') >= sample_min_reads)
                .filter(
                    pl.col('umi').is_in(pl.col('umi').sample(sample_n_umis))
                )    
                .collect(streaming=True)
            )

            if df.shape[0] == 0:
                return None, f"No UMIs found with at least {sample_min_reads} reads"
        else:
            df = pl.read_parquet(file_path)

        return df, file_path
    except Exception as e:
        return None, f"Error loading data: {str(e)}"
