# Data processing functions
import polars as pl
from pathlib import Path
import ogtk.ltr.align as al
import logging

logger = logging.getLogger("fracture_app.data_processing")

def get_umis(df):
    """Extract unique UMIs from the data frame"""
    return df.select('umi').unique().to_series().to_list()

def get_selected_umi_stats(df, umi):
    """Get statistics for a selected UMI"""
    if df is None or not umi:
        return "No UMI selected"
    if umi not in df['umi'].unique():
        return "No valid UMI selected"
    
    try:
        umi_reads = df.filter(pl.col('umi') == umi).height
        return umi_reads
    except Exception as e:
        logger.error(f"Error in selected_umi_stats: {e}")
        return "Error calculating UMI stats"

def compute_coverage(df, umi, ref_str, max_range=None):
    """Compute coverage of reads for a UMI"""
    try:
        logger.debug(f"compute_coverage called with umi={umi}, ref_str_len={len(ref_str)}")
        
        # Check if UMI exists in data
        available_umis = df.select('umi').unique().to_series().to_list()
        if umi not in available_umis:
            raise ValueError(f"UMI '{umi}' not found in data. Available UMIs: {len(available_umis)}")
            
        filtered_df = (
            df
            .filter(pl.col('umi') == umi)
            .with_columns(intbc=pl.lit('in'))  # dummy intbc field needed for alignment
        )
        
        logger.debug(f"Filtered data has {len(filtered_df)} rows for UMI {umi}")
        
        if len(filtered_df) == 0:
            raise ValueError(f"No data found for UMI '{umi}' after filtering")
            
        # Check required columns
        required_cols = ['umi', 'r2_seq', 'intbc']  # assuming r2_seq is the sequence column
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {filtered_df.columns}")
        
        logger.debug(f"Calling al.compute_coverage with max_range={max_range or len(ref_str)}")
        result = al.compute_coverage(filtered_df, ref_str, max_range=max_range or len(ref_str))
        
        logger.debug(f"Coverage computation successful, result shape: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error in compute_coverage: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise

def assemble_umi(df, target_umi, k, min_coverage, method, auto_k=False, 
                export_graphs=True, only_largest=True, 
                start_anchor=None, end_anchor=None, min_length=None):
    """Assemble a contig for a specific UMI"""
    result = (
        df
        .pp.assemble_umi(
            target_umi=target_umi,
            k=int(k),
            min_coverage=int(min_coverage),
            method=method,
            auto_k=auto_k,
            export_graphs=export_graphs,
            only_largest=only_largest,
            start_anchor=start_anchor,
            min_length=min_length,
            end_anchor=end_anchor,
        )
    )
    return result

def sweep_assembly_params(df, target_umi, start_anchor, end_anchor, 
                         k_start, k_end, k_step, cov_start, cov_end, cov_step, 
                         method, min_length=None, export_graphs=False, prefix=""):
    """Run parameter sweep for assembly"""
    result = (
        df.pp.sweep_assembly_params(
            target_umi=target_umi,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            k_start=k_start,
            k_end=k_end,
            k_step=k_step,
            cov_start=cov_start,
            cov_end=cov_end,
            cov_step=cov_step,
            method=method,
            min_length=min_length,
            export_graphs=export_graphs,
            prefix=f"{prefix}{target_umi}_"
        )
        .pivot(
            values="contig_length",
            index="k",
            columns="min_coverage"
        )
    )
    
    return result
