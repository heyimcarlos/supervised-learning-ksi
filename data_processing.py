"""
Toronto KSI Collisions Analysis and Modelling

This script performs data exploration and modeling on the Toronto KSI (Killed or Seriously Injured) collisions dataset.

Authors: Carlos De La Cruz, Manav, Harsh, and Rishi
Date: 2023-10-31
"""

"""
1. Data exploration: a complete review and analysis of the dataset including:

Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. – use pandas, numpy and any other python packages.
Statistical assessments including means, averages, correlations
Missing data evaluations – use pandas, numpy and any other python packages
Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any other python packages, you also can use power BI desktop.


2. Data modelling:

Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
Feature selection – use pandas and sci-kit learn. (The group needs to justify each feature used and any data columns discarded)
Train, Test data splitting – use numpy, sci-kit learn.
Managing imbalanced classes if needed. Check here for info: https://elitedatascience.com/imbalanced-classes
Use pipelines class to streamline all the pre-processing transformations.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.core.base import np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

DATA_FILE = "ksi.csv"
PLOTS_DIR = "plots"
DATASET_DIR = "datasets"


def load_data(file_path: str):
    """Loads the df from the specified file path."""
    if not os.path.exists(file_path):
        print(f"Error: df not found at {file_path}")
        return None
    df = pd.read_csv(file_path)
    return df


def investigate_dataset(df: pd.DataFrame):
    """
    Performs an initial investigation of the dataset.

    Args:
        df (pd.DataFrame): The dataset to investigate.
    """
    print("\ndf Info:")
    df.info()

    print("\ndf Description:")
    print(df.describe())

    print("\ndf Shape:")
    print(df.shape)

    print("\ndf Data Types:")
    print(df.dtypes)

    print("\ndf Head:")
    print(df.head(3))


# --- Data Cleaning and Preprocessing ---


def handle_missing_accident_numbers(df):
    """
    Handles missing 'accident_number' values using a loop-based approach
    by grouping records based on a composite key.

    Args:
        df (pd.DataFrame): The DataFrame with 'accident_number' and other relevant columns.

    Returns:
        pd.DataFrame: The DataFrame with 'accident_number' column imputed.
    """
    df_processed = df.copy()

    # Ensure 'accident_number' can hold string values for new IDs
    # and properly treat existing NaNs.
    df_processed["accident_number"] = df_processed["accident_number"].astype(str)
    df_processed["accident_number"] = df_processed["accident_number"].replace(
        "nan", np.nan
    )

    # Define the columns that form the composite key
    composite_key_cols = [
        "date",
        "time",
        "street1",
        "latitude",
        "longitude",
        "light",
        "accident_class",
    ]

    # --- Pre-processing for the loop ---
    # Convert composite key columns to string for consistent hashing/comparison in a loop
    # and fill NaNs with a unique string placeholder.
    # This prevents NaN values from breaking comparisons.
    temp_df = df_processed[composite_key_cols].astype(str).fillna("__MISSING__")

    # Store unique composite keys and their assigned accident numbers
    # Key: Tuple of composite column values (e.g., (date_val, time_val, ...))
    # Value: The assigned accident_number for that composite key
    composite_key_to_accnum = {}

    # A counter for generating new unique accident numbers
    new_acc_id_counter = 0

    for index, row in df_processed.iterrows():
        current_acc_num = row["accident_number"]

        if pd.notna(current_acc_num):
            # Also, ensure this existing accident number is mapped to its composite key
            # for later use if other rows without accnum share this key
            current_composite_key_values = tuple(temp_df.loc[index, composite_key_cols])
            if current_composite_key_values not in composite_key_to_accnum:
                composite_key_to_accnum[current_composite_key_values] = current_acc_num
            continue  # Move to the next row

        # If accident_number is NaN, we need to determine its value
        current_composite_key_values = tuple(temp_df.loc[index, composite_key_cols])

        # Check if this composite key has already been seen and assigned an accident_number
        if current_composite_key_values in composite_key_to_accnum:
            # If yes, assign the existing accident_number to the current row
            df_processed.loc[index, "accident_number"] = composite_key_to_accnum[
                current_composite_key_values
            ]
        else:
            # If no, this is a new unique composite key (a new accident)
            # Assign a new unique accident number and store it
            new_id = new_acc_id_counter + 1
            df_processed.loc[index, "accident_number"] = new_id
            composite_key_to_accnum[current_composite_key_values] = new_id
            new_acc_id_counter += 1

    return df_processed


def clean_data(df):
    """
    Performs data cleaning and preprocessing steps.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    df_cleaned = df.copy()

    # Column renames mapping
    column_renames = {
        "ACCNUM": "accident_number",
        "ACCLOC": "accident_location",
        "ACCLASS": "accident_class",
        "RDSFCOND": "road_surface_condition",
        "IMPACTYPE": "impact_type",
        "INVTYPE": "involvement_type",
        "INVAGE": "involvement_age",
        "VEHTYPE": "vehicle_type",
        "DRIVACT": "driver_action",
        "DRIVCOND": "driver_condition",
        "PEDTYPE": "pedestrian_type",
        "PEDACT": "pedestrian_action",
        "PEDCOND": "pedestrian_condition",
        "CYCLISTYPE": "cyclist_type",
        "CYCACT": "cyclist_action",
        "CYCCOND": "cyclist_condition",
        "AG_DRIV": "aggressive_driving",
        "INITDIR": "initial_direction",
        "NEIGHBOURHOOD_158": "neighbourhood_new",
        "HOOD_158": "hood_id_new",
        "NEIGHBOURHOOD_140": "neighbourhood_old",
        "HOOD_140": "hood_id_old",
    }

    # Apply renames
    df_cleaned.rename(columns=column_renames, inplace=True)
    # Lowercase all column names
    df_cleaned.columns = [col.lower() for col in df_cleaned.columns]

    # --- pre-aggregation ---

    #  Remove 6 samples with accident_class 'Property Damage 0'
    df_cleaned = df_cleaned[df_cleaned["accident_class"] != "Property Damage O"]

    # Convert 'DATE' to datetime
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")
    df_cleaned["year"] = df_cleaned["date"].dt.year

    # Create temp accident numbers for null samples
    # nan_accident_numbers = df_cleaned["accident_number"].isnull()
    # df_cleaned.loc[nan_accident_numbers, "accident_number"] = np.arange(
    #     1, nan_accident_numbers.sum() + 1
    # )  # Start numbering from 1 since existing numbers are very large
    df_cleaned = handle_missing_accident_numbers(df_cleaned)

    # Convert boolean columns to 1/0. Assume NaN are 0.
    boolean_columns = [
        "pedestrian",
        "cyclist",
        "automobile",
        "motorcycle",
        "truck",
        "trsn_city_veh",
        "emerg_veh",
        "passenger",
        "speeding",
        "aggressive_driving",
        "redlight",
        "alcohol",
        "disability",
    ]
    for col in boolean_columns:
        if col in df_cleaned.columns:
            # df_cleaned.replace({col: {"Yes": 1, "No": 0}}, inplace=True)
            # .apply is better because it takes 'null' into consideration
            df_cleaned[col] = df_cleaned[col].apply(lambda x: 1 if x == "Yes" else 0)

    injury_mapping = {
        "Fatal": 4,
        "Major": 3,
        "Minor": 2,
        "Minimal": 1,
        "None": 0,
    }
    # Apply mapping
    df_cleaned["injury_severity_score"] = df_cleaned["injury"].map(injury_mapping)

    # --- aggregation ---

    def get_most_frequent(series):
        return series.mode()[0] if not series.mode().empty else np.nan

    def get_all_unique(series):
        return list(series.dropna().unique())

    def get_all_values(series):
        return list(series.dropna())

    #  TODO: Aggregation strategies
    # - Group involvement ages into different bins. Maybe we just use the actually involved? like the drivers?
    aggregation_dict = {
        "accident_number": "first",
        # Unique accident identifiers
        "date": "first",
        "time": "first",
        "year": "first",
        "street1": "first",
        "street2": "first",
        "offset": "first",
        "road_class": "first",
        "district": "first",
        "latitude": "first",
        "longitude": "first",
        "x": "first",
        "y": "first",
        "accident_location": "first",
        "traffctl": "first",
        "visibility": "first",
        "light": "first",
        "road_surface_condition": "first",
        "hood_id_new": "first",
        "neighbourhood_new": "first",
        "hood_id_old": "first",
        "neighbourhood_old": "first",
        "accident_class": "first",
        # Person-level counts/booleans
        # booleans don't get toggled by person. If an accident involved a pedestrian,
        # all samples are marked as "yes".
        "pedestrian": "first",
        "cyclist": "first",
        "automobile": "first",
        "motorcycle": "first",
        "truck": "first",
        "trsn_city_veh": "first",
        "emerg_veh": "first",
        "passenger": "first",
        "speeding": "first",
        "aggressive_driving": "first",
        "redlight": "first",
        "alcohol": "first",
        "disability": "first",
        # Person Characteristic
        "impact_type": get_most_frequent,
        "involvement_type": get_all_unique,
        "initial_direction": get_most_frequent,
        "vehicle_type": get_all_unique,
        "manoeuver": get_all_unique,
        "driver_action": get_all_unique,
        "driver_condition": get_all_unique,
        "pedestrian_type": get_all_unique,
        "pedestrian_action": get_all_unique,
        "pedestrian_condition": get_all_unique,
        "cyclist_type": get_all_unique,
        "cyclist_action": get_all_unique,
        "cyclist_condition": get_all_unique,
        "division": "first",
        #  NOTE: adds data leakage.
        # "injury": lambda x: "Fatal" in x.astype(str).values,
        # numerical averages
        "involvement_age": get_all_values,
        #  NOTE: adds data leakage.
        "injury_severity_score": "max",
    }
    aggregated_df = (
        df_cleaned.groupby("accident_number")
        .agg(aggregation_dict)
        .reset_index(drop=True)
    )

    # after aggregation categorizing involvement_age call
    aggregated_df = categorize_age_groups(aggregated_df)

    #  NOTE: add is_weekend feature
    aggregated_df["is_weekend"] = (
        aggregated_df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    )

    #  NOTE: add season feature
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
    aggregated_df["season"] = aggregated_df["date"].dt.month.map(month_to_season)

    #  NOTE: Create the 'time_of_day' feature
    hour = aggregated_df["time"] // 100  # Convert time to hour (e.g., 1450 -> 14)
    # Bins: Night(0-6), Morning(7-9), Day(10-15), Evening(16-18), Night(19-23)
    bins = [-1, 6, 9, 15, 18, 23]
    labels = ["Night", "Morning Rush", "Daytime", "Evening Rush", "Night"]
    aggregated_df["time_of_day"] = pd.cut(hour, bins=bins, labels=labels, ordered=False)

    # --- post-aggregation ---
    #  INFO: Investigation
    # Get accidents where pedestrian_action has multiple values
    # accident_number = 886 has ['Crossing, no Traffic Control', 'Crossing without right of way']
    # so this is a multi-label feature.
    # multi_label_pedestrian_actions = aggregated_df[
    #     aggregated_df["pedestrian_action"].apply(lambda x: len(x) > 1)
    # ]
    # print(df_cleaned["pedestrian_type"].unique())

    # --- missing values handling ---

    #  TODO: rewrite the aggregation to return null instead of an empty list.

    #  TODO: driver_action and driver_condition have a lot of missin values, in the shape of empty lists
    # sometimes a cyclist crashes into a parked car, so we can assume that the driver was not involved.
    # investigate further

    #  INFO: Impute multilabel features with 'Not Applicable' if the list is empty.
    for multi_label_feature in [
        "pedestrian_action",
        "pedestrian_condition",
        "pedestrian_type",
        "cyclist_action",
        "cyclist_condition",
        "cyclist_type",
    ]:
        if multi_label_feature in aggregated_df.columns:
            aggregated_df[multi_label_feature] = aggregated_df[
                multi_label_feature
            ].apply(
                lambda x: (
                    ["Not Applicable"] if isinstance(x, list) and len(x) == 0 else x
                )
            )

    #  INFO: Impute single label features with 'Unknown' if the list is empty.
    for unknown_feature in [
        "driver_action",
        "driver_condition",
        "vehicle_type",
        "manoeuver",
    ]:
        if unknown_feature in aggregated_df.columns:
            aggregated_df[unknown_feature] = aggregated_df[unknown_feature].apply(
                lambda x: (["Unknown"] if isinstance(x, list) and len(x) == 0 else x)
            )

    # Impute accident_location with 'Unknown'
    aggregated_df.fillna({"accident_location": "Unknown"}, inplace=True)

    # Impute road_class with the most frequent value for street1 and street2
    street1_road_class_map = (
        aggregated_df.groupby("street1")["road_class"]
        .apply(get_most_frequent)
        .to_dict()
    )
    mask_missing_road_class = aggregated_df["road_class"].isnull()
    for i, row in aggregated_df[mask_missing_road_class].iterrows():
        street = row["street1"]
        if street in street1_road_class_map and pd.notna(
            street1_road_class_map[street]
        ):
            aggregated_df.loc[i, "road_class"] = street1_road_class_map[street]

    mask_remaining_missing_road_class = aggregated_df["road_class"].isnull()
    if mask_remaining_missing_road_class.any():
        street2_road_class_map = (
            aggregated_df.groupby("street2")["road_class"]
            .apply(get_most_frequent)
            .to_dict()
        )
        for i, row in aggregated_df[mask_remaining_missing_road_class].iterrows():
            street = row["street2"]
            if street in street2_road_class_map and pd.notna(
                street2_road_class_map[street]
            ):
                aggregated_df.loc[i, "road_class"] = street2_road_class_map[street]

    # Impute district the district value where street1 or street2 matches
    street1_district_map = (
        aggregated_df.groupby("street1")["district"].apply(get_most_frequent).to_dict()
    )
    mask_missing_district = aggregated_df["district"].isnull()
    for i, row in aggregated_df[mask_missing_district].iterrows():
        street = row["street1"]
        if street in street1_district_map and pd.notna(street1_district_map[street]):
            aggregated_df.loc[i, "district"] = street1_district_map[street]

    mask_remaining_missing_district = aggregated_df["district"].isnull()
    if mask_remaining_missing_district.any():
        street2_district_map = (
            aggregated_df.groupby("street2")["district"]
            .apply(get_most_frequent)
            .to_dict()
        )
        for i, row in aggregated_df[mask_remaining_missing_district].iterrows():
            street = row["street2"]
            if street in street2_district_map and pd.notna(
                street2_district_map[street]
            ):
                aggregated_df.loc[i, "district"] = street2_district_map[street]

    # Fill any remaining missing district with 'Unknown'
    aggregated_df.fillna({"district": "Unknown"}, inplace=True)

    # Fill any remaining missing road_class with 'Unknown'
    aggregated_df.fillna({"road_class": "Unknown"}, inplace=True)

    # Impute street2 with n/a
    aggregated_df.fillna({"street2": "Not Applicable"}, inplace=True)

    # Impute traffctl with 'Missing control'
    aggregated_df.fillna({"traffctl": "Missing Control"}, inplace=True)

    #  NOTE: missing values are so low for visibility, light and road_surface_condition that's not worth it to cross impute them.

    # Impute visibility with 'Unknown'
    aggregated_df.fillna({"visibility": "Unknown"}, inplace=True)

    # Impute light with 'Unknown'
    aggregated_df.fillna({"light": "Unknown"}, inplace=True)

    # Impute road_surface_condition with 'Unknown'
    aggregated_df.fillna({"road_surface_condition": "Unknown"}, inplace=True)

    # Impute initial_direction with 'Unknown'
    aggregated_df.fillna({"initial_direction": "Unknown"}, inplace=True)

    # Impute involvement_type with 'Unknown' because only 9 samples are missing
    aggregated_df.fillna({"impact_type": "Unknown"}, inplace=True)

    # Impute accident_class with 'Fatal' is injury_severity_score is 4
    aggregated_df.loc[aggregated_df["injury_severity_score"] == 4, "accident_class"] = (
        "Fatal"
    )

    #  NOTE:
    # You wanted to create features indicating the main involvement ages (e.g., "driver and pedestrian"). With these count-based features, you don't need to create new flags. The model can learn these interactions directly.
    # For example, when your model sees a row where pedestrian = 1, automobile = 1, and age_25_59 = 2, it has all the information it needs to learn the pattern for an accident involving a pedestrian and a driver in the 25-59 age range. The combination of your boolean involvement types (pedestrian, cyclist, etc.) and your numerical age counts is extremely effective.

    # --- Drop features ---
    columns_to_drop = []

    # Drop `offset ` because it is not useful for analysis and it's missing 79% of the time.
    if "offset" in aggregated_df.columns:
        columns_to_drop.append("offset")
    # aggregated_df.drop(columns=["offset"], inplace=True)

    #  NOTE: injury_severity_score adds data leakage, we need to remove it, to avoid our model generalizing that value 4 = fatal.
    if "injury_severity_score" in aggregated_df.columns:
        columns_to_drop.append("injury_severity_score")
    # aggregated_df.drop(columns=["injury_severity_score"], inplace=True)

    #  NOTE: drop involvement_type because we already have a list of boolean features, rendering it redundant.
    if "involvement_type" in aggregated_df.columns:
        columns_to_drop.append("involvement_type")
    # aggregated_df.drop(columns=["involment_type"], inplace=True)

    # INFO: Drop pedestrian_type because it's a detail of how the pedestrian was involved (we cannot use a sentence.
    # decided to keep it for now, since we multilabel imputed.
    # aggregated_df.drop(columns=["pedestrian_type"], inplace=True)

    #  NOTE: drop latitude and longitude because we already have x and y (in cardinal distance) coordinates.
    if "latitude" in aggregated_df.columns:
        columns_to_drop.append("latitude")

    if "longitude" in aggregated_df.columns:
        columns_to_drop.append("longitude")

    #  NOTE: drop neighbourhood_old and hood_id_old because they use an older naming system.
    if "neighbourhood_old" in aggregated_df.columns:
        columns_to_drop.append("neighbourhood_old")

    if "hood_id_old" in aggregated_df.columns:
        columns_to_drop.append("hood_id_old")

    # NOTE: drop division because it means the toronto police division, which is not useful for analysis.
    if "division" in aggregated_df.columns:
        columns_to_drop.append("division")
    # aggregated_df.drop(columns=["division"], inplace=True)

    #  TODO: the features below are to be visited in the future, they might be helpful.

    #  NOTE: drop hood_id_new and neighbourhood_new because they are not useful for analysis.
    if "hood_id_new" in aggregated_df.columns:
        columns_to_drop.append("hood_id_new")

    if "neighbourhood_new" in aggregated_df.columns:
        columns_to_drop.append("neighbourhood_new")

    #  NOTE: drop street1 and street2 because we already X and Y coordinates.
    if "street1" in aggregated_df.columns:
        columns_to_drop.append("street1")

    if "street2" in aggregated_df.columns:
        columns_to_drop.append("street2")

    #  NOTE: drop district because it is not useful for analysis.
    if "district" in aggregated_df.columns:
        columns_to_drop.append("district")

    # The original 'time' column is also redundant now.
    if "time" in aggregated_df.columns:
        columns_to_drop.append("time")
    if "date" in aggregated_df.columns:
        columns_to_drop.append("date")
    if "year" in aggregated_df.columns:
        columns_to_drop.append("year")

    if "accident_number" in aggregated_df.columns:
        columns_to_drop.append("accident_number")

    aggregated_df.drop(columns=columns_to_drop, inplace=True)

    return aggregated_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    # INFO: pre-encoding clean up
    # multilabel features
    pedestrian_action_mappings = {
        "Not Applicable": "not_applicable",
        "Crossing with right of way": "crossing_with_right_of_way",
        "Crossing, Pedestrian Crossover": "crossing_pedestrian_crossover",
        "Other": "other",
        "On Sidewalk or Shoulder": "on_sidewalk_or_shoulder",
        "Crossing, no Traffic Control": "crossing_no_traffic_control",
        "Running onto Roadway": "running_onto_roadway",
        "Crossing without right of way": "crossing_without_right_of_way",
        "Coming From Behind Parked Vehicle": "coming_from_behind_parked_vehicle",
        "Person Getting on/off Vehicle": "person_getting_on_off_vehicle",
        "Walking on Roadway with Traffic": "walking_on_roadway_with_traffic",
        "Playing or Working on Highway": "playing_working_on_highway",
        "Crossing marked crosswalk without ROW": "crossing_marked_crosswalk_without_row",
        "Walking on Roadway Against Traffic": "walking_on_roadway_against_traffic",
        "Pushing/Working on Vehicle": "pushing_working_on_vehicle",
        "Person Getting on/off School Bus": "person_getting_on_off_school_bus",
    }

    pedestrian_condition_mappings = {
        "Inattentive": "inattentive",
        "Normal": "normal",
        "Not Applicable": "not_applicable",
        "Medical or Physical Disability": "medical_or_physical_disability",
        "Unknown": "unknown",
        "Other": "other",
        "Had Been Drinking": "alcohol_involved",
        "Ability Impaired, Alcohol": "alcohol_involved",
        "Ability Impaired, Alcohol Over .80": "alcohol_involved",
        "Ability Impaired, Drugs": "ability_impaired_drugs",
        "Fatigue": "fatigue",
    }

    pedestrian_type_mappings = {
        "Not Applicable": "not_applicable",
        "Vehicle turns left while ped crosses with ROW at inter.": (
            "vehicle_turn_left_ped_with_row"
        ),
        "Vehicle turns left while ped crosses without ROW at inter.": (
            "vehicle_turn_left_ped_without_row"
        ),
        "Vehicle turns right while ped crosses with ROW at inter.": (
            "vehicle_turn_right_ped_with_row"
        ),
        "Vehicle turns right while ped crosses without ROW at inter.": (
            "vehicle_turn_right_ped_without_row"
        ),
        "Vehicle is going straight thru inter.while ped cross with ROW": (
            "vehicle_straight_ped_with_row"
        ),
        "Vehicle is going straight thru inter.while ped cross without ROW": (
            "vehicle_straight_ped_without_row"
        ),
        "Pedestrian hit a PXO/ped. Mid-block signal": "pedestrian_hit_on_roadside",
        "Pedestrian hit on sidewalk or shoulder": "pedestrian_hit_on_roadside",
        "Pedestrian hit at mid-block": "pedestrian_hit_on_roadside",
        "Pedestrian hit at private driveway": "pedestrian_hit_on_driveway",
        "Pedestrian involved in a collision with transit vehicle anywhere along roadway": (
            "pedestrian_collision_transit_vehicle"
        ),
        "Vehicle hits the pedestrian walking or running out from between parked vehicles at mid-block": (
            "vehicle_hits_pedestrian_between_parked_vehicles"
        ),
        "Other / Undefined": "other_undefined",
        "Pedestrian hit at parking lot": "pedestrian_hit_on_parking_lot",
        "Unknown": "unknown",
    }

    cyclist_type_mappings = {
        "Motorist turns right at non-signal Inter.(stop, yield, no cont.,and dwy) and strikes cyclist.": (
            "motorist_action_right_turn"
        ),
        "Motorist turning right on green or amber at signalized intersection strikes cyclist.": (
            "motorist_action_right_turn"
        ),
        "Motorist turning right on red at signalized intersection strikes cyclist.": (
            "motorist_action_right_turn"
        ),
        "Cyclist and Driver travelling in same direction. One vehicle rear-ended the other.": (
            "collision_same_direction"
        ),
        "Cyclist and Driver travelling in same direction. One vehicle sideswipes the other.": (
            "collision_same_direction"
        ),
        "Insufficient information (to determine cyclist crash type).": "insufficient_info",
        "Cyclist without ROW rides into path of motorist at inter, lnwy, dwy-Cyclist not turn.": (
            "cyclist_action_failed_to_yield"
        ),
        "Not Applicable": "not_applicable",
        "Cyclist turned left across motorists path.": "cyclist_action_improper_turn",
        "Cyclist turns right across motorists path": "cyclist_action_improper_turn",
        "Cyclist makes u-turn in-front of driver.": "cyclist_action_improper_turn",
        "Motorist makes u-turn in-front of cyclist.": "motorist_action_u_turn",
        "Motorist turned left across cyclists path.": "motorist_action_left_turn",
        "Cyclist strikes a parked vehicle.": "collision_stationary_object",
        "Cyclist falls off bike - no contact with motorist.": "cyclist_action_loses_control",
        "Motorist loses control and strikes cyclist.": "motorist_action_loses_control",
        "Cyclist struck opened vehicle door": "collision_stationary_object",
        "Motorist without ROW drives into path of cyclist at inter, lnwy, dwy-Driver not turn.": (
            "motorist_action_failed_to_yield"
        ),
        "Motorist reversing struck cyclist.": "motorist_action_reversing",
        "Cyclist loses control and strikes object (pole, ttc track)": "cyclist_action_loses_control",
        "Cyclist strikes pedestrian.": "collision_stationary_object",
        "Cyclist struck at PXO(cyclist either travel in same dir. as veh. or ride across xwalk)": (
            "cyclist_action_roadside_interaction"
        ),
        "Cyclist rode off sidewalk into road at midblock.": "cyclist_action_roadside_interaction",
    }

    cyclist_condition_mappings = {
        "Not Applicable": "not_applicable",
        "Had Been Drinking": "alcohol_involved",
        "Unknown": "unknown",
        "Normal": "normal",
        "Inattentive": "inattentive",
        "Ability Impaired, Alcohol": "alcohol_involved",
        "Other": "other",
        "Ability Impaired, Alcohol Over .80": "alcohol_involved",
        "Medical or Physical Disability": "medical_or_physical_disability",
        "Ability Impaired, Drugs": "ability_impaired_drugs",
        "Fatigue": "fatigue",
    }

    cyclist_action_mappings = {
        "Not Applicable": "not_applicable",
        "Improper Turn": "improper_maneuver",
        "Improper Lane Change": "improper_maneuver",
        "Improper Passing": "improper_maneuver",
        "Disobeyed Traffic Control": "disregard_traffic_laws",
        "Failed to Yield Right of Way": "disregard_traffic_laws",
        "Wrong Way on One Way Road": "disregard_traffic_laws",
        "Driving Properly": "driving_properly",
        "Other": "other",
        "Lost control": "reckless_operation",
        "Speed too Fast For Condition": "reckless_operation",
        "Following too Close": "reckless_operation",
    }

    driver_action_mappings = {
        "Driving Properly": "driving_properly",
        "Failed to Yield Right of Way": "failed_to_yield",
        "Lost control": "reckless_operation",
        "Improper Turn": "improper_maneuver",
        "Improper Passing": "improper_maneuver",
        "Improper Lane Change": "improper_maneuver",
        "Other": "other",
        "Disobeyed Traffic Control": "disregard_traffic_laws",
        "Following too Close": "reckless_operation",
        "Exceeding Speed Limit": "reckless_operation",
        "Speed too Fast For Condition": "reckless_operation",
        "Unknown": "unknown",
        "Speed too Slow": "reckless_operation",
        "Wrong Way on One Way Road": "disregard_traffic_laws",
    }

    driver_condition_mappings = {
        "Normal": "normal",
        "Inattentive": "inattentive",
        "Unknown": "unknown",
        "Medical or Physical Disability": "medical_or_physical_disability",
        "Had Been Drinking": "alcohol_involved",
        "Other": "other",
        "Ability Impaired, Alcohol Over .08": "alcohol_involved",
        "Ability Impaired, Alcohol": "alcohol_involved",
        "Fatigue": "fatigue",
        "Ability Impaired, Drugs": "ability_impaired_drugs",
    }

    vehicle_type_mappings = {
        "Automobile, Station Wagon": "passenger_vehicle",
        "Passenger Van": "passenger_vehicle",
        "Taxi": "taxi",
        "Pick Up Truck": "truck",
        "Truck - Closed (Blazer, etc)": "truck",
        "Truck-Tractor": "truck",
        "Truck - Dump": "truck",
        "Truck (other)": "truck",
        "Truck - Car Carrier": "truck",
        "Truck - Tank": "truck",
        "Truck - Open": "truck",
        "Delivery Van": "truck",
        "Tow Truck": "truck",
        "Municipal Transit Bus (TTC)": "bus",
        "Bus (Other) (Go Bus, Gray Coa": "bus",
        "Intercity Bus": "bus",
        "School Bus": "bus",
        "Other Emergency Vehicle": "emergency_vehicle",
        "Police Vehicle": "emergency_vehicle",
        "Ambulance": "emergency_vehicle",
        "Fire Vehicle": "emergency_vehicle",
        "Motorcycle": "motorcycle_moped",
        "Moped": "motorcycle_moped",
        "Construction Equipment": "off_road_or_construction",
        "Off Road - 2 Wheels": "off_road_or_construction",
        "Off Road - 4 Wheels": "off_road_or_construction",
        "Off Road - Other": "off_road_or_construction",
        "Bicycle": "bicycle",
        "Street Car": "street_car",
        "Other": "other_unknown",
        "Unknown": "unknown",
        "Rickshaw": "other_unknown",
    }

    manoeuver_mappings = {
        "Going Ahead": "going_ahead",
        "Overtaking": "going_ahead",
        "Turning Left": "turning",
        "Turning Right": "turning",
        "Making U Turn": "turning",
        "Changing Lanes": "position_change",
        "Merging": "position_change",
        "Pulling Away from Shoulder or Curb": "position_change",
        "Pulling Onto Shoulder or towardCurb": "position_change",
        "Reversing": "position_change",
        "Slowing or Stopping": "stationary_or_slowing",
        "Stopped": "stationary_or_slowing",
        "Parked": "stationary_or_slowing",
        "Disabled": "stationary_or_slowing",
        "Other": "other_unknown",
        "Unknown": "other_unknown",
    }

    mappings = [
        ("pedestrian_action", pedestrian_action_mappings),
        ("pedestrian_condition", pedestrian_condition_mappings),
        ("pedestrian_type", pedestrian_type_mappings),
        ("cyclist_action", cyclist_action_mappings),
        ("cyclist_condition", cyclist_condition_mappings),
        ("cyclist_type", cyclist_type_mappings),
        ("driver_action", driver_action_mappings),
        ("driver_condition", driver_condition_mappings),
        ("vehicle_type", vehicle_type_mappings),
        ("manoeuver", manoeuver_mappings),
    ]

    for col, mapping in mappings:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda actions: [mapping.get(action, "other") for action in actions]
            )

    for col, _ in mappings:
        # print(f"Encoding multi-label column: {col}")
        if col in df.columns:
            mlb = MultiLabelBinarizer()

            # Fit and transform the column
            encoded_data = mlb.fit_transform(df[col])

            # Create a new DataFrame with the encoded columns
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=[f"{col}_{c.lower().replace(' ', '_')}" for c in mlb.classes_],
            )

            # Join the new DataFrame with the original one
            df = df.join(encoded_df)

            # Drop the original multi-label column
            df = df.drop(col, axis=1)

    # INFO: single-label features
    single_label = [
        "impact_type",
        "initial_direction",
        "road_surface_condition",
        # "light",
        "traffctl",
        "visibility",
        "road_class",
        "season",
        "time_of_day",
        "accident_location",
    ]

    #  INFO: add manual mappings for light conditions
    light_mappings = {
        "Daylight": "daylight",
        "Dark": "dark",
        "Dark, artificial": "dark_artificial",
        "Dawn": "dawn",
        "Dawn, artificial": "dawn_artificial",
        "Dusk": "dusk",
        "Dusk, artificial": "dusk_artificial",
        "Other": "other",
        "Unknown": "unknown",
    }
    df["light"] = df["light"].map(light_mappings)
    df = pd.get_dummies(
        df,
        columns=["light"],
        prefix="light",
        prefix_sep="_",
        dtype="int",
    )

    # one-hot encode single-label features
    df = pd.get_dummies(
        df,
        columns=single_label,
        prefix=single_label,
        prefix_sep="_",
        drop_first=True,
        dtype="int",
    )

    #  INFO: New features

    coords = df[["x", "y"]]
    # eps: max distance between two samples
    #  INFO: check the coordinate_clustering function to determine the best eps value.
    db = DBSCAN(eps=300, min_samples=10).fit(coords)
    # -1 indicates noise (outliers not in any cluster)
    df["accident_hotspot_cluster"] = db.labels_

    # INFO: bad weather and low light
    df["bad_weather_and_dark"] = (
        (
            (df["road_surface_condition_Wet"] == 1)
            | (df["road_surface_condition_Ice"] == 1)
        )
        & (df["light_dark"] == 1)
    ).astype(int)

    # # INFO: Reckless at intersection
    df["reckless_at_intersection"] = (
        (df["driver_action_reckless_operation"] == 1)
        & (df["accident_location_Intersection Related"] == 1)
    ).astype(int)

    return df


def perform_data_quality_check(df_cleaned):
    """
    Performs a detailed investigation of the df after cleaning.
    Check for data consistency, redundancy, missing values, and relationships
    between features.

    Args:
        df_cleaned (pd.DataFrame): The cleaned df.
    """

    # accident_number as string
    df_cleaned["accident_number"] = df_cleaned["accident_number"].astype("str")

    accnum_missing_count = df_cleaned["accident_number"].isnull().sum()
    print(f"Accident Number Missing Count: {accnum_missing_count}\n")

    # Group by accident number
    accident_counts = df_cleaned.groupby("accident_number").size()
    print(f"{accident_counts}\n")
    print(f"Number of unique non-null accidents: {accident_counts.shape[0]}\n")
    print(f"Max individuals in an accident: {accident_counts.max()}\n")
    print(f"Min individuals in an accident: {accident_counts.min()}\n")

    # Check for inconsistent accident_class values
    acclass_consistency = df_cleaned.groupby("accident_number")[
        "accident_class"
    ].nunique()
    # inconsistent_accidents: accidents with more than one unique accident_class (e.g., Fatal, Non-Fatal)
    inconsistent_accidents = acclass_consistency[acclass_consistency > 1]
    print(
        f"Number of accidents with inconsistent accident_class values: {inconsistent_accidents.shape[0]}"
    )

    # Check redundancy between (x,y) and (latitude, longitude)
    corr_x_long = df_cleaned["x"].corr(df_cleaned["longitude"])
    corr_y_lat = df_cleaned["y"].corr(df_cleaned["latitude"])
    print(f"Correlation between x and Longitude: {corr_x_long}")
    print(f"Correlation between y and Latitude: {corr_y_lat}")

    # Identify features with high missing values and low utility
    missing_percentages = df_cleaned.isnull().sum() / len(df_cleaned) * 100
    highly_missing_features = missing_percentages[missing_percentages > 50].sort_values(
        ascending=False
    )
    print("\nFeatures with more than 50% missing values:")
    print(highly_missing_features)

    # Check relationship between 'accident_class' and 'injury' for 'Fatal' cases
    fatal_acclass_df = df_cleaned[df_cleaned["accident_class"] == "Fatal"]
    print(
        "\nFatal accident class count:",
        fatal_acclass_df["accident_number"].value_counts(),
    )
    fatal_injury_per_fatal_acclass = fatal_acclass_df.groupby("accident_number")[
        "injury"
    ].apply(lambda x: "Fatal" in x.values)
    print("\nfatal_injury_per_fatal_acclass:\n", fatal_injury_per_fatal_acclass)
    print(
        f"\nNumber of 'Fatal' accident_numbers where at least one injury is 'Fatal': {fatal_injury_per_fatal_acclass.sum()}"
    )

    # Check if all Fatal accident_numbers have at least one 'Fatal' injury
    all_fatal_acclass_have_fatal_injury = fatal_injury_per_fatal_acclass.all()
    print(
        f"Do all 'Fatal' accident_numbers have at least one 'Fatal' INJURY? {all_fatal_acclass_have_fatal_injury}"
    )


def coordinate_clustering(df):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Assuming 'df' is your DataFrame
    coords = df[["x", "y"]]

    # 1. Calculate the distance to the 10th nearest neighbor for each point
    # (The value for n_neighbors should be the same as your min_samples)
    k = 10
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(coords)
    distances, indices = neighbors_fit.kneighbors(coords)

    # 2. Sort the distances and plot them
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]  # We only need the distance to the k-th neighbor

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f"K-Distance Graph (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}-th nearest neighbor")
    plt.grid(True)
    plt.show()


# Data visualization
def data_visualisation(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # --- Data Preparation for Temporal Plots ---
    # Convert 'DATE' to datetime objects
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Extract Year, Month, Day of Week, and Hour for easier plotting
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month_name()
    df["DAY_OF_WEEK"] = df["DATE"].dt.day_name()
    # Convert integer time (e.g., 1450) to just the hour (e.g., 14)
    df["HOUR"] = df["TIME"] // 100

    # --- Graph 1: Accident Severity Breakdown ---
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="ACCLASS", palette="viridis")
    plt.title("Distribution of Accident Severity")
    plt.xlabel("Accident Class")
    plt.ylabel("Number of Persons Involved")
    plt.savefig(r"graphs/1_countplot_acclass.png")
    plt.close()

    # --- Graph 2: Collisions by Light Condition ---
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        y="LIGHT",
        order=df["LIGHT"].value_counts().index,
        palette="magma",
    )
    plt.title("Number of Collisions by Light Condition")
    plt.xlabel("Number of Persons Involved")
    plt.ylabel("Light Condition")
    plt.tight_layout()
    plt.savefig(r"graphs/2_countplot_light.png")
    plt.close()

    # --- Graph 3: Collisions by Impact Type ---
    plt.figure(figsize=(12, 8))
    sns.countplot(
        data=df,
        y="IMPACTYPE",
        order=df["IMPACTYPE"].value_counts().index,
        palette="plasma",
    )
    plt.title("Number of Collisions by Impact Type")
    plt.xlabel("Number of Persons Involved")
    plt.ylabel("Impact Type")
    plt.tight_layout()
    plt.savefig(r"graphs/3_countplot_impactype.png")
    plt.close()

    # --- Graph 4: Severity by Road Class (Percentage) ---
    crosstab_roadclass = (
        pd.crosstab(df["ROAD_CLASS"], df["ACCLASS"], normalize="index") * 100
    )
    crosstab_roadclass.plot(
        kind="bar", stacked=True, figsize=(12, 8), colormap="coolwarm"
    )
    plt.title("Proportion of Accident Severity by Road Class")
    plt.xlabel("Road Class")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Accident Class")
    plt.tight_layout()
    plt.savefig(r"graphs/4_stacked_severity_by_roadclass.png")
    plt.close()

    # --- Graph 5: Severity by Contributing Factors (Alcohol) ---
    plt.figure(figsize=(10, 7))
    sns.countplot(
        data=df[df["ALCOHOL"] == "Yes"],
        x="ALCOHOL",
        hue="ACCLASS",
        palette="Reds_r",
    )
    plt.title("Accident Severity in Alcohol-Related Collisions")
    plt.xlabel("Alcohol Involved")
    plt.ylabel("Number of Persons Involved")
    plt.legend(title="Accident Class")
    plt.tight_layout()
    plt.savefig(r"graphs/5_grouped_severity_alcohol.png")
    plt.close()

    # --- Graph 6: Severity vs. Aggressive/Distracted Driving ---
    plt.figure(figsize=(10, 7))
    sns.countplot(
        data=df[df["AG_DRIV"] == "Yes"],
        x="AG_DRIV",
        hue="ACCLASS",
        palette="Oranges_r",
    )
    plt.title("Accident Severity in Aggressive/Distracted Driving Collisions")
    plt.xlabel("Aggressive/Distracted Driving Involved")
    plt.ylabel("Number of Persons Involved")
    plt.legend(title="Accident Class")
    plt.tight_layout()
    plt.savefig(r"graphs/6_grouped_severity_ag_driv.png")
    plt.close()

    # --- Graph 7: Collisions Over the Years ---
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x="YEAR", hue="ACCLASS", palette="viridis")
    plt.title("KSI Collisions by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Persons Involved")
    plt.xticks(rotation=45)
    plt.legend(title="Accident Class")
    plt.tight_layout()
    plt.savefig(r"graphs/7_line_collisions_by_year.png")
    plt.close()

    # --- Graph 8: Collisions by Hour of Day ---
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x="HOUR", palette="twilight_shifted")
    plt.title("Number of Collisions by Hour of Day")
    plt.xlabel("Hour of Day (24-hour format)")
    plt.ylabel("Number of Persons Involved")
    plt.tight_layout()
    plt.savefig(r"graphs/8_bar_collisions_by_hour.png")
    plt.close()

    # --- Graph 7: Collisions Over the Years ---
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x="YEAR", hue="ACCLASS", palette="viridis")
    plt.title("KSI Collisions by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Persons Involved")
    plt.xticks(rotation=45)
    plt.legend(title="Accident Class")
    plt.tight_layout()
    plt.savefig(r"graphs/7_line_collisions_by_year.png")
    plt.close()

    # --- Graph 8: Collisions by Hour of Day ---
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x="HOUR", palette="twilight_shifted")
    plt.title("Number of Collisions by Hour of Day")
    plt.xlabel("Hour of Day (24-hour format)")
    plt.ylabel("Number of Persons Involved")
    plt.tight_layout()
    plt.savefig(r"graphs/8_bar_collisions_by_hour.png")
    plt.close()

    # --- Graph 9: Collisions by Day of Week ---
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="DAY_OF_WEEK", order=days_order, palette="cubehelix")
    plt.title("Number of Collisions by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Persons Involved")
    plt.tight_layout()
    plt.savefig(r"graphs/9_bar_collisions_by_day.png")
    plt.close()

    # --- Graph 10: Collision Hotspot Map ---
    plt.figure(figsize=(10, 10))
    sns.kdeplot(
        data=df,
        x="LONGITUDE",
        y="LATITUDE",
        fill=True,
        cmap="Reds",
        alpha=0.6,
        levels=20,  # Increase levels for more detail
    )
    plt.title("Collision Hotspot Density in Toronto")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(r"graphs/10_kdeplot_hotspots.png")
    plt.close()


# age vs fatality analysis (call commented out in main)
def analyze_age_groups(df):
    from collections import defaultdict

    # Create a dictionary to store age ranges and their fatality rates
    age_fatality_data = defaultdict(lambda: {"total": 0, "fatal": 0})

    # Process each accident
    for idx, row in df.iterrows():
        age_ranges = row["involvement_age"]  # This is already a list from your cleaning
        is_fatal = row["accident_class"] == "Fatal"

        for age_range in age_ranges:
            if pd.notna(age_range):
                age_fatality_data[age_range]["total"] += 1
                if is_fatal:
                    age_fatality_data[age_range]["fatal"] += 1

    # Calculate fatality rates
    age_ranges = []
    fatality_rates = []
    total_counts = []

    for age_range, counts in age_fatality_data.items():
        if counts["total"] > 0:  # Avoid division by zero
            age_ranges.append(age_range)
            fatality_rates.append((counts["fatal"] / counts["total"]) * 100)
            total_counts.append(counts["total"])

    # Sort by age range
    sorted_indices = sorted(
        range(len(age_ranges)),
        key=lambda k: int(age_ranges[k].split("-")[0]) if "-" in age_ranges[k] else 0,
    )
    age_ranges = [age_ranges[i] for i in sorted_indices]
    fatality_rates = [fatality_rates[i] for i in sorted_indices]
    total_counts = [total_counts[i] for i in sorted_indices]

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Plot 1: Fatality rates by age group
    plt.subplot(2, 1, 1)
    plt.bar(age_ranges, fatality_rates, color="red", alpha=0.6)
    plt.title("Fatality Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Fatality Rate (%)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 2: Total incidents by age group
    plt.subplot(2, 1, 2)
    plt.bar(age_ranges, total_counts, color="blue", alpha=0.6)
    plt.title("Total Incidents by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("age_analysis.png")
    plt.close()

    # Print statistical summary
    print("\nAge Group Analysis Summary:")
    print("----------------------------------------")
    for age_range, rate, count in zip(age_ranges, fatality_rates, total_counts):
        print(f"Age Group: {age_range}")
        print(f"Fatality Rate: {rate:.2f}%")
        print(f"Total Incidents: {count}")
        print("----------------------------------------")

    return age_fatality_data


# cleaning and splitting involvement_age
def categorize_age_groups(df):
    def count_ages_in_groups(age_list):
        age_groups = {
            "0-14": 0,
            "15-24": 0,
            "25-59": 0,
            "60-74": 0,
            "75+": 0,
            "total": 0,
        }

        if not isinstance(age_list, list):
            return pd.Series(age_groups)

        for age_range in age_list:
            if pd.isna(age_range):
                continue

            if age_range == "Over 95":
                age_groups["75+"] += 1
                continue

            try:

                start_age = int(age_range.split()[0])
                age_groups["total"] += 1

                if start_age <= 14:
                    age_groups["0-14"] += 1
                elif start_age <= 24:
                    age_groups["15-24"] += 1
                elif start_age <= 59:
                    age_groups["25-59"] += 1
                elif start_age <= 74:
                    age_groups["60-74"] += 1
                else:
                    age_groups["75+"] += 1
            except (ValueError, IndexError):
                continue

        return pd.Series(age_groups)

    # apply categorization
    age_group_counts = df["involvement_age"].apply(count_ages_in_groups)

    # adding new columns to dataframe based on the class divide decided
    df[
        [
            "age_0_14",
            "age_15_24",
            "age_25_59",
            "age_60_74",
            "age_75_plus",
            #  NOTE: Include "total_involved" to keep track of total individuals involved in the accident
            "total_involved",
        ]
    ] = age_group_counts

    # drop involvement_age after this
    df.drop("involvement_age", axis=1, inplace=True)

    return df


if __name__ == "__main__":
    # 1. Load data
    ksi_df = load_data(os.path.join(DATASET_DIR, DATA_FILE))

    if ksi_df is not None:
        # 2. Initial data investigation
        # investigate_dataset(ksi_df)

        # 3. Clean data
        df = clean_data(ksi_df)

        # 3.1 age analysis for deciding the class divide for age, will plot a graph, as well write fatality for each class in og dataset (0-4, 5-9, 10-14 etctetc)
        # age_analysis = analyze_age_groups(cleaned_df)

        # 4. Perform data quality check
        # perform_data_quality_check(cleaned_df)

        # 5. Feature engineering (or perform encoding, imputing, drop features)
        #  TODO: https://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection
        # Use feature importances (of the selected classifier) for feature selection.
        df = feature_engineering(df)

        # 6. Save the cleaned df's
        X = df.drop(columns=["accident_class"])
        y = df["accident_class"]

        # --- THIS IS THE ONLY PLACE YOU SPLIT THE DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # --- SAVE THE FILES ---
        X_train.to_csv("datasets/X_train.csv", index=False)
        X_test.to_csv("datasets/X_test.csv", index=False)
        y_train.to_csv("datasets/y_train.csv", index=False)
        y_test.to_csv("datasets/y_test.csv", index=False)

        # Export cleaned df to CSV
        now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        # for windows user (most)
        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        df.to_csv(os.path.join(DATASET_DIR, f"toronto_ksi_{now}.csv"), index=False)
