"""
Feature Engineering Module for Airline Flight Operations.
Enriches flight operations with turnaround physics, passenger Special Service Requests (SSR),
baggage logistics metrics, network hub congestion, and pre-departure features.
"""

from typing import Tuple
import pandas as pd
import numpy as np

from src.config import (
    ENRICHED_COMPLEXITY_PATH,
    PRIMARY_KEYS,
)
from src.data_loader import load_raw_data, load_processed_summaries
from src.utils import get_logger

logger = get_logger(__name__)

def build_enriched_dataset() -> pd.DataFrame:
    """
    Construct the master enriched flight complexity dataset by integrating:
    1. Base operational complexity summary (8,099 flights)
    2. Raw flight schedule datetimes, fleet type, seats, and minimum turn times
    3. PNR Special Service Requests (Airport Wheelchairs, Electric Chairs, Unaccompanied Minors)
    4. Airport international flags and hub congestion rates
    5. Turnaround buffer physics and load factor metrics
    6. Objective FAA operational delay labels (>15 min delay)

    Returns:
        pd.DataFrame: Enriched master dataframe saved to data/processed/enriched_flight_complexity.csv
    """
    logger.info("Building enriched flight operations dataset...")

    # Load raw and processed tables
    raw_flights, airports, pnr_remarks = load_raw_data()
    _, _, _, overall_df = load_processed_summaries()

    df = overall_df.copy()

    # 1. Integrate raw flight operational attributes
    # The raw_flights and overall_df share identical 8,099 rows in 1-to-1 order
    meta_cols = [
        "total_seats",
        "fleet_type",
        "carrier",
        "scheduled_ground_time_minutes",
        "actual_ground_time_minutes",
        "minimum_turn_minutes",
        "scheduled_departure_datetime_local",
        "scheduled_arrival_datetime_local",
        "actual_departure_datetime_local",
        "actual_arrival_datetime_local",
    ]
    for col in meta_cols:
        if col in raw_flights.columns:
            df[col] = raw_flights[col].values

    # 2. Extract and Aggregate Special Service Requests (SSR) from PNR Remarks
    logger.info("Aggregating Special Service Requests (SSR) by flight number...")
    ssr_pivot = pnr_remarks.pivot_table(
        index="flight_number",
        columns="special_service_request",
        aggfunc="size",
        fill_value=0,
    ).reset_index()

    # Standardize SSR column names
    col_mapping = {
        "Airport Wheelchair": "ssr_airport_wheelchair",
        "Electric Wheelchair": "ssr_electric_wheelchair",
        "Manual Wheelchair": "ssr_manual_wheelchair",
        "Unaccompanied Minor": "ssr_unaccompanied_minor",
    }
    for orig, standard in col_mapping.items():
        if orig in ssr_pivot.columns:
            ssr_pivot.rename(columns={orig: standard}, inplace=True)
        else:
            ssr_pivot[standard] = 0

    ssr_cols = list(col_mapping.values())
    ssr_pivot["ssr_total"] = ssr_pivot[ssr_cols].sum(axis=1)

    # Merge SSR features onto master dataframe
    df = pd.merge(df, ssr_pivot[["flight_number"] + ssr_cols + ["ssr_total"]], on="flight_number", how="left")
    for c in ssr_cols + ["ssr_total"]:
        df[c] = df[c].fillna(0).astype(int)

    # SSR rate per 100 passengers
    df["ssr_per_100_pax"] = np.where(df["total_pax"] > 0, (df["ssr_total"] / df["total_pax"]) * 100.0, 0.0)

    # 3. Airport & International Route Features
    logger.info("Deriving airport network and international route flags...")
    airports_dedup = airports.drop_duplicates("airport_iata_code")
    air_map = dict(zip(airports_dedup["airport_iata_code"], airports_dedup["iso_country_code"]))

    df["origin_country"] = df["scheduled_departure_station_code"].map(air_map).fillna("Unknown")
    df["dest_country"] = df["scheduled_arrival_station_code"].map(air_map).fillna("Unknown")
    df["is_international"] = (
        (df["origin_country"] != df["dest_country"]) & (df["dest_country"] != "Unknown")
    ).astype(int)

    # 4. Temporal & Schedule Features
    dep_dt = pd.to_datetime(df["scheduled_departure_datetime_local"], errors="coerce")
    df["scheduled_dep_hour"] = dep_dt.dt.hour.fillna(12).astype(int)
    df["scheduled_dep_dayofweek"] = dep_dt.dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["scheduled_dep_dayofweek"].isin([5, 6]).astype(int)

    # Categorize time of day
    def assign_time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    df["departure_time_of_day"] = df["scheduled_dep_hour"].apply(assign_time_of_day)

    # 5. Aircraft Capacity & Turnaround Squeeze Physics
    # Load factor: actual passengers / total aircraft seats
    df["load_factor"] = np.where(
        df["total_seats"] > 0,
        np.clip(df["total_pax"] / df["total_seats"], 0.0, 1.5),
        0.80,
    )

    # Baggage per passenger
    df["bags_per_pax"] = np.where(
        df["total_pax"] > 0,
        df["total_bags"].fillna(0) / df["total_pax"],
        0.0,
    )

    # Turnaround Buffer = Scheduled Ground Time - Minimum Required Turn Time
    df["buffer_minutes"] = df["scheduled_ground_time_minutes"] - df["minimum_turn_minutes"]

    # Turnaround Tightness Ratio = Minimum Turn / Scheduled Ground Time
    # (Values close to or > 1 indicate severe schedule squeeze / zero operational cushion)
    df["turnaround_tightness_ratio"] = np.where(
        df["scheduled_ground_time_minutes"] > 0,
        df["minimum_turn_minutes"] / df["scheduled_ground_time_minutes"],
        1.0,
    )

    # 6. Station & Route Congestion Metrics
    logger.info("Computing station and route historical congestion rates...")
    # Origin station average departure delay
    origin_delay = df.groupby("scheduled_departure_station_code")["departure_delay"].mean().to_dict()
    df["origin_avg_delay"] = df["scheduled_departure_station_code"].map(origin_delay).fillna(0.0)

    # Destination station average arrival delay
    dest_delay = df.groupby("scheduled_arrival_station_code")["arrival_delay"].mean().to_dict()
    df["dest_avg_delay"] = df["scheduled_arrival_station_code"].map(dest_delay).fillna(0.0)

    # Route flight frequency
    route_counts = df["route"].value_counts().to_dict()
    df["route_flight_count"] = df["route"].map(route_counts).fillna(1).astype(int)

    # 7. Clean missing values in baggage columns (only 4 rows had missing total_bags)
    bag_impute_cols = [
        "total_bags",
        "num_origin_bags",
        "num_transfer_bags",
        "num_hot_transfer_bags",
        "transfer_ratio",
        "hot_transfer_ratio",
    ]
    for c in bag_impute_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 8. Define Objective Operational Ground-Truth Targets
    # FAA standard: delayed if departure delay > 15 min or arrival delay > 15 min
    df["delayed_15min_flag"] = (
        (df["departure_delay"] > 15) | (df["arrival_delay"] > 15)
    ).astype(int)

    # Severe operational complexity flag (Hard class)
    df["high_complexity_flag"] = (df["complexity_class"] == "Hard").astype(int)

    # Save to processed directory
    df.to_csv(ENRICHED_COMPLEXITY_PATH, index=False)
    logger.info(f"Enriched flight dataset successfully saved to: {ENRICHED_COMPLEXITY_PATH}")
    logger.info(f"Master Dataset Shape: {df.shape}")

    return df

if __name__ == "__main__":
    enriched_df = build_enriched_dataset()
    print("Class distribution:\n", enriched_df["complexity_class"].value_counts())
    print("Delayed > 15 min rate:", enriched_df["delayed_15min_flag"].mean())
    print("High complexity rate:", enriched_df["high_complexity_flag"].mean())
