"""
Data Loader and Validation Module.
Handles ingestion of raw airline operations data, schema validation,
missing data audits, and standardized preprocessing.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import (
    RAW_FLIGHTS_PATH,
    RAW_AIRPORTS_PATH,
    RAW_PNR_REMARKS_PATH,
    SUMMARY_FLIGHT_PATH,
    SUMMARY_PASSENGER_PATH,
    SUMMARY_BAG_PATH,
    OVERALL_COMPLEXITY_PATH,
    PRIMARY_KEYS,
)
from src.utils import get_logger

logger = get_logger(__name__)

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw flight-level operational data, airports country registry, and PNR remarks.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (flights_df, airports_df, pnr_remarks_df)
    """
    logger.info("Loading raw datasets from data/raw/...")
    flights_df = pd.read_csv(RAW_FLIGHTS_PATH)
    airports_df = pd.read_csv(RAW_AIRPORTS_PATH)
    pnr_remarks_df = pd.read_csv(RAW_PNR_REMARKS_PATH)

    logger.info(f"Loaded Raw Flights: {flights_df.shape}")
    logger.info(f"Loaded Airports: {airports_df.shape}")
    logger.info(f"Loaded PNR Remarks: {pnr_remarks_df.shape}")

    return flights_df, airports_df, pnr_remarks_df

def load_processed_summaries() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-aggregated summary tables across flight, passenger, and bag levels.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (flight_summary, passenger_summary, bag_summary, overall_complexity)
    """
    logger.info("Loading processed summary tables from data/processed/...")
    flight_summary = pd.read_csv(SUMMARY_FLIGHT_PATH)
    passenger_summary = pd.read_csv(SUMMARY_PASSENGER_PATH)
    bag_summary = pd.read_csv(SUMMARY_BAG_PATH)
    overall_complexity = pd.read_csv(OVERALL_COMPLEXITY_PATH)

    logger.info(f"Flight Summary: {flight_summary.shape}")
    logger.info(f"Passenger Summary: {passenger_summary.shape}")
    logger.info(f"Bag Summary: {bag_summary.shape}")
    logger.info(f"Overall Complexity: {overall_complexity.shape}")

    return flight_summary, passenger_summary, bag_summary, overall_complexity

def audit_data_quality(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Perform a comprehensive data quality check:
    - Missing value counts and percentages
    - Duplicate records
    - Numeric ranges and memory usage
    """
    total_rows = len(df)
    missing_series = df.isna().sum()
    missing_cols = missing_series[missing_series > 0]
    duplicates = df.duplicated().sum()

    audit_report = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "duplicate_rows": int(duplicates),
        "missing_features_count": len(missing_cols),
        "missing_breakdown": {col: int(cnt) for col, cnt in missing_cols.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    logger.info(f"Audit [{dataset_name}]: {total_rows} rows, {duplicates} duplicate rows, "
                f"{len(missing_cols)} columns with missing values.")
    return audit_report
