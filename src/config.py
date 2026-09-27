"""
Configuration Module for Flight Complexity Analysis and Prediction System.
Centralizes all file paths, random seeds, target definitions, and feature schemas.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
for path in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Raw Data Files
RAW_FLIGHTS_PATH = RAW_DATA_DIR / "Flight_Level_Data.csv"
RAW_AIRPORTS_PATH = RAW_DATA_DIR / "Airports_Data.csv"
RAW_PNR_REMARKS_PATH = RAW_DATA_DIR / "PNR_Remark_Level_Data.csv"

# Processed Data Files
SUMMARY_FLIGHT_PATH = PROCESSED_DATA_DIR / "flight_level_summary.csv"
SUMMARY_PASSENGER_PATH = PROCESSED_DATA_DIR / "flight_passenger_summary.csv"
SUMMARY_BAG_PATH = PROCESSED_DATA_DIR / "flight_bag_summary.csv"
OVERALL_COMPLEXITY_PATH = PROCESSED_DATA_DIR / "overall_complexity.csv"
ENRICHED_COMPLEXITY_PATH = PROCESSED_DATA_DIR / "enriched_flight_complexity.csv"

# Global Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Primary Keys used for joining flight operations
PRIMARY_KEYS = [
    "company_id",
    "flight_number",
    "scheduled_departure_date_local",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
]

# Baseline Rule-Based Weights
RULE_BASED_WEIGHTS = {
    "delay_complexity_score": 0.25,
    "passenger_complexity_score": 0.50,
    "bag_complexity_score": 0.25,
}

# Target Column Definitions
TARGET_MULTICLASS = "complexity_class"
TARGET_CONTINUOUS = "overall_flight_complexity"
TARGET_DELAY_15MIN = "delayed_15min_flag"
TARGET_HIGH_COMPLEXITY = "high_complexity_flag"

# Pre-Departure Predictive Feature Space (Strictly T-2 hours before departure)
PRE_DEPARTURE_NUMERIC_FEATURES = [
    "total_seats",
    "scheduled_ground_time_minutes",
    "minimum_turn_minutes",
    "buffer_minutes",
    "turnaround_tightness_ratio",
    "total_pax",
    "load_factor",
    "child_ratio",
    "basic_economy_ratio",
    "stroller_ratio",
    "total_bags",
    "bags_per_pax",
    "transfer_ratio",
    "hot_transfer_ratio",
    "num_hot_transfer_bags",
    "ssr_total",
    "ssr_airport_wheelchair",
    "ssr_unaccompanied_minor",
    "origin_avg_delay",
    "dest_avg_delay",
    "route_flight_count",
    "scheduled_dep_hour",
    "scheduled_dep_dayofweek",
    "is_weekend",
    "is_international",
]

PRE_DEPARTURE_CATEGORICAL_FEATURES = [
    "company_id",
    "carrier",
    "fleet_type",
    "departure_time_of_day",
]

# Full Operational Feature Space (Including post-flight turnaround and delay metrics)
FULL_NUMERIC_FEATURES = PRE_DEPARTURE_NUMERIC_FEATURES + [
    "departure_delay",
    "arrival_delay",
    "flightfly_delay",
    "ground_time_utilization",
    "pressure_on_staff",
    "ground_time_deviation",
    "turnaround_risk",
    "delay_complexity_score",
    "passenger_complexity_score",
    "bag_complexity_score",
]

FULL_CATEGORICAL_FEATURES = PRE_DEPARTURE_CATEGORICAL_FEATURES
