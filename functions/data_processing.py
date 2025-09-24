"""
Data Processing Functions for SVI Fraud Detection
Standalone functions extracted from notebooks for data preprocessing and daily pipelines
"""

from pyspark.sql.functions import (
    col, row_number, greatest, least, collect_list, lower, mean, mode, when, 
    regexp_replace, min, max, datediff, to_date, concat, lit, round, 
    date_format, hour, udf, size, array, expr, dayofweek, month
)
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql import Window
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import logging


# ============================================================================
# Data Preprocessing Functions (from notebooks/DataPreprocessing.py)
# ============================================================================

def get_referral_vertices(df): 
    """Process claim referral log and create risk indicators"""
    df = df.withColumn("Claim Ref", regexp_replace("Claim Ref", "\\*", "")) \
        .withColumn("siu_investigated", when(col("Source of referral") == "SIU",1)
                                     .otherwise(0))
    
    # Create indicator fraud investigation risk
    risk_cols = {"Final Outcome of Claim": ["Withdrawn whilst investigation ongoing", "Repudiated – Litigated – Claim then discontinued", "Repudiated – Litigated – Success at trial", "Repudiated – Not challenged"]
    }
    risk_cols["Outcome of Referral"] = ["Accepted"]
    risk_cols["Outcome of investigation"] = ["Repudiated", "Repudiated in part", "Under Investigation", "Withdrawn whilst investigation ongoing"]

    for this_col in risk_cols:
        df = df.withColumn(f'{this_col}_risk', 
                          col(this_col).isin(*risk_cols[this_col]).cast('integer')) 

    df = df.fillna({"Final Outcome of Claim_risk": 0, 
                    "Outcome of Referral_risk": 0, 
                    "Outcome of investigation_risk": 0})
                                     
    df = df.withColumn("fraud_risk", greatest("Final Outcome of Claim_risk", "Outcome of Referral_risk", "Outcome of investigation_risk"))

    referral_vertices = df.select(
        col("Claim Ref").alias("id"), 
        "siu_investigated", 
        "fraud_risk", "Final Outcome of Claim_risk", 
        "Outcome of Referral_risk", "Outcome of investigation_risk",
        col("Concerns").alias("referral_concerns"),
        col("Date received").alias("transact_time"),
        col("Date received").alias("referral_log_date"),
        col("Date of Outcome").alias("referral_outcome_date")
    )
    return referral_vertices


def calculate_damage_score(*args):
    """Calculate damage score and damage area counts"""
    damageScore = 1
    areasDamagedMinimal = 0
    areasDamagedMedium = 0
    areasDamagedHeavy = 0
    areasDamagedSevere = 0
    
    for damage in args:
        if damage == 'Minimal':
            damageScore *= 2
            areasDamagedMinimal += 1
        elif damage == 'Medium':
            damageScore *= 3
            areasDamagedMedium += 1
        elif damage == 'Heavy':
            damageScore *= 4
            areasDamagedHeavy += 1
        elif damage == 'Severe':
            damageScore *= 5
            areasDamagedSevere += 1
    
    return damageScore, areasDamagedMinimal, areasDamagedMedium, areasDamagedHeavy, areasDamagedSevere


def create_check_variables(df):
    """Generate check variables C1-C14 for fraud detection"""
    
    # C1: was the incident on a Friday/Saturday *NIGHT*?
    df = df.withColumn("incident_day_of_week", date_format(col("latest_event_time"), "E"))
    
    fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | 
                     (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4))))
                                                                                                                        
    df = df.withColumn(
        "C1_fri_sat_night",
        when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0))

    df = df.withColumn("reported_day_of_week", date_format(col("latest_event_time"), "E"))

    # C2: Was there a delay in notifying us of the incident without reason?
    df = df.withColumn("delay_in_reporting", datediff(col("reported_date"), col("start_date")))
    df = df.withColumn("C2_reporting_delay", when(col("delay_in_reporting")>=3, 1).when(col("delay_in_reporting").isNull(), 1).otherwise(0))

    # Add a column to check if the incident date is on a weekend
    df = df.withColumn(
        "is_incident_weekend",
        when(date_format(col("start_date"), "E").isin("Fri", "Sat", "Sun"), 1).otherwise(0)
    )

    # Add a column to check if the reported date is on a Monday
    df = df.withColumn(
        "is_reported_monday",
        when(date_format(col("reported_date"), "E") == "Mon", 1).otherwise(0)
    )

    # C3: Cases taking place over a weekend but not being reported until Monday
    df = df.withColumn(
        "C3_weekend_incident_reported_monday",
        when((col("is_incident_weekend") == True) & (col("is_reported_monday") == True), 1).otherwise(0)
    )

    # C5: Incident between 11pm and 5am
    df = df.withColumn(
        "C5_is_night_incident",
        when((hour(col("start_date")) >= 23) | (hour(col("start_date")) <= 5) | (hour(col("start_date"))).isNull(), 1).otherwise(0)
    )

    # C6: No commuting on policy and customer travelling between the hours of 6am and 10am or 3pm and 6pm?
    not_commuting_rush = (lower(col("vehicle_use_quote")) == 1) & ((hour(col("start_date")).between(6, 10)) | (hour(col("start_date")).between(15, 18)))
    df = df.withColumn(
        "C6_no_commuting_but_rush_hour",
        when(not_commuting_rush, 1).when(not_commuting_rush.isNull(), 1).otherwise(0)
    )

    # C7: Notified of a incident/CRN from the PH relating to the police attending the scene? (low risk)
    df = df.withColumn(
        "C7_police_attended_or_crime_reference",
        when((col("is_police_attendance") == True) | (col("is_crime_reference_provided") == True), 1).otherwise(0)
    )

    # C9: Was the policy incepted within 30 days of the incident date?
    df = df.withColumn("inception_to_claim", datediff(to_date(col("start_date")), to_date(col("policy_start_date"))))
    
    df = df.withColumn(
        "C9_policy_within_30_days",
        when(col("inception_to_claim").between(0, 30),1).when(col("inception_to_claim").isNull(), 1).otherwise(0)
    )

    # C10: Does the policy end within 1 or 2 months of the incident date?
    df = df.withColumn("claim_to_policy_end", datediff(to_date(col("policy_renewal_date")), to_date(col("start_date"))))
    
    df = df.withColumn(
        "C10_claim_to_policy_end",
            when(col("claim_to_policy_end")<60, 1). when(col("claim_to_policy_end").isNull(), 1).otherwise(0)
            )

    df = df.withColumn( "driver_age_low_1", when(col("age_at_policy_start_date_1")<25, 1)
                                   .when(col("age_at_policy_start_date_1").isNull(), 1).otherwise(0)
                                   )
    df = df.withColumn( "claim_driver_age_low", when(col("min_claim_driver_age")<25, 1)
                                   .when(col("min_claim_driver_age").isNull(), 1).otherwise(0))

    # Check licence low threshold
    df = df.withColumn( "licence_low_1", when(col("licence_length_years_1")<=3, 1).otherwise(0))

    # C11: Are they classed as young/inexperienced ie under 25 or new to driving
    condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1) 
    df = df.withColumn( "C11_young_or_inexperienced", when(condition_inexperienced, 1)
                                   .when(condition_inexperienced.isNull(), 1)
                                   .otherwise(0))

    # C12: Age in comparison to the type of vehicle (Value wise). Thresholds by business unit
    condition_expensive_car =  ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | ( (col("age_at_policy_start_date_1") >= 25) &(col("vehicle_value") >= 30000))
    
    df = df.withColumn( "C12_expensive_for_driver_age", when(condition_expensive_car, 1)
                        .when(condition_expensive_car.isNull(), 1)
                        .otherwise(0))

    # Create a regex pattern from the watch words
    watch_words = "|".join(["commut", "deliver", "parcel", "drink", "police", "custody", "arrest", 
                            "alcohol", "drug", "station", "custody"])

    # Add a column to check if Circumstances contains any of the items in list
    df = df.withColumn(
        "C14_contains_watchwords",
        when(lower(col("Circumstances")).rlike(watch_words), 1)
        .when(col("Circumstances").isNull(), 1).otherwise(0)
    )
    
    return df


# ============================================================================
# Daily Policies Functions (from notebooks/daily_pipeline/Daily_Policies.py)
# ============================================================================

def get_latest_policy_transactions(spark):
    """Get the latest transaction for each policy"""
    query = """
    WITH latest_trans AS (
        SELECT DISTINCT 
            FIRST_VALUE(pt.policy_number) OVER(PARTITION BY pt.policy_number ORDER BY pt.updated_date DESC) AS policy_number,
            FIRST_VALUE(pt.policy_transaction_id) OVER(PARTITION BY pt.policy_number ORDER BY pt.updated_date DESC) AS policy_transaction_id
        FROM prod_adp_certified.policy_motor.policy_transaction pt
    )
    SELECT DISTINCT * FROM latest_trans
    """
    return spark.sql(query)


def get_policy_data(spark):
    """Retrieve comprehensive policy information"""
    query = """
    SELECT DISTINCT 
        pt.policy_transaction_id,
        pt.policy_number,
        pt.policy_start_date,
        pt.policy_renewal_date,
        pt.policy_type,
        MAX(qi.session_id) as quote_session_id,
        MAX(qi.quote_iteration_id) as quote_iteration_id 
    FROM prod_adp_certified.policy_motor.policy_transaction pt
    LEFT JOIN prod_adp_certified.quote_motor.quote_iteration qi
    ON pt.quote_iteration_id = qi.quote_iteration_id
    GROUP BY pt.policy_transaction_id, pt.policy_number, pt.policy_start_date, 
             pt.policy_renewal_date, pt.policy_type
    """
    return spark.sql(query)


def get_vehicle_data(spark):
    """Retrieve vehicle information from policy and quote tables"""
    query = """
    SELECT DISTINCT
        pt.policy_transaction_id,
        v.vehicle_use_code as vehicle_use_quote,
        v.year_of_manufacture,
        v.registration_date,
        v.car_group,
        v.vehicle_value,
        v.purchase_date,
        pv.overnight_location_abi_code,
        pv.vehicle_overnight_location_id,
        pv.vehicle_overnight_location_name,
        pv.business_mileage,
        pv.annual_mileage
    FROM prod_adp_certified.policy_motor.policy_transaction pt
    LEFT JOIN prod_adp_certified.quote_motor.quote_iteration qi
    ON pt.quote_iteration_id = qi.quote_iteration_id
    LEFT JOIN prod_adp_certified.quote_motor.vehicle v
    ON qi.quote_iteration_id = v.quote_iteration_id
    LEFT JOIN prod_adp_certified.policy_motor.vehicle pv
    ON pt.policy_transaction_id = pv.policy_transaction_id
    """
    return spark.sql(query)


def get_excess_data(spark):
    """Retrieve excess and premium information"""
    query = """
    SELECT DISTINCT
        pt.policy_transaction_id,
        cpv.voluntary_amount,
        pcp.policyholder_ncd_years,
        pcp.ncd_protected_flag,
        pt.sales_channel
    FROM prod_adp_certified.policy_motor.policy_transaction pt
    LEFT JOIN prod_adp_certified.policy_motor.comprehensive_policy_vehicle cpv
    ON pt.policy_transaction_id = cpv.policy_transaction_id
    LEFT JOIN prod_adp_certified.policy_motor.policy_car_products pcp
    ON pt.policy_transaction_id = pcp.policy_transaction_id
    """
    return spark.sql(query)


def get_driver_data(spark):
    """Retrieve driver information for all drivers on policy"""
    query = """
    SELECT DISTINCT
        pt.policy_transaction_id,
        pd.position as driver_position,
        pd.date_of_birth,
        pd.additional_vehicles_owned,
        pd.age_at_policy_start_date,
        pd.cars_in_household,
        pd.licence_length_years,
        pd.years_resident_in_uk,
        pd.employment_type_abi_code,
        pz.postcode
    FROM prod_adp_certified.policy_motor.policy_transaction pt
    LEFT JOIN prod_adp_certified.policy_motor.driver pd
    ON pt.policy_transaction_id = pd.policy_transaction_id
    LEFT JOIN prod_adp_certified.quote_motor.post_code pz
    ON pt.policy_transaction_id = pz.policy_transaction_id
    """
    return spark.sql(query)


def get_svi_claims(spark):
    """Retrieve SVI performance data with risk assessments"""
    query = """
    SELECT DISTINCT
        `Claim Number` as claim_number,
        `Result of Outsourcing` as tbg_outcome,
        `FA Outcome` as fa_outcome,
        CASE 
            WHEN lower(`Result of Outsourcing`) = 'settled' THEN 0 
            WHEN lower(`Result of Outsourcing`) IN ('withdrawn', 'repudiated', 'managed away', 'cancelled') THEN 1
        END AS tbg_risk,
        CASE 
            WHEN lower(`FA Outcome`) IN ('claim closed', 'claim to review', 'not comprehensive cover') THEN 1 
            ELSE 0 
        END AS fa_risk,
        `Notification Date` as notification_date
    FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance
    WHERE `Notification Date` >= '2023-01-01'
    AND (lower(`Result of Outsourcing`) IS NULL OR 
         lower(`Result of Outsourcing`) NOT IN ('ongoing - client', 'ongoing - tbg', 'pending closure'))
    AND lower(`FA Outcome`) != 'not comprehensive cover'
    """
    return spark.sql(query)


def get_latest_claim_versions(spark):
    """Get latest claim versions for SVI claims"""
    query = """
    WITH target_claims AS (
        SELECT DISTINCT
            `Claim Number` as claim_number,
            `Result of Outsourcing` as tbg_outcome,
            `FA Outcome` as fa_outcome,
            CASE 
                WHEN lower(`Result of Outsourcing`) = 'settled' THEN 0 
                WHEN lower(`Result of Outsourcing`) IN ('withdrawn', 'repudiated', 'managed away', 'cancelled') THEN 1
            END AS tbg_risk,
            CASE 
                WHEN lower(`FA Outcome`) IN ('claim closed', 'claim to review', 'not comprehensive cover') THEN 1 
                ELSE 0 
            END AS fa_risk
        FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance
        WHERE `Notification Date` >= '2023-01-01'
    )
    SELECT DISTINCT
        MAX(cv.claim_number) AS claim_number,
        MAX(tc.tbg_risk) AS tbg_risk,
        MAX(tc.fa_outcome) AS fa_outcome,
        MAX(tc.fa_risk) AS fa_risk,
        MAX(cv.claim_version_id) AS claim_version_id,
        cv.claim_id,
        MAX(cv.event_enqueued_utc_time) AS latest_event_time
    FROM target_claims tc
    LEFT JOIN prod_adp_certified.claim.claim_version cv
    ON LOWER(cv.claim_number) = LOWER(tc.claim_number)
    GROUP BY cv.claim_id
    HAVING claim_number IS NOT NULL
    """
    return spark.sql(query)


# ============================================================================
# Daily Claims Functions (from notebooks/daily_pipeline/Daily_Claims.py)
# ============================================================================

def get_latest_claim_version(spark, date_str):
    """Get latest claim version data for specified date"""
    query = f"""
    SELECT DISTINCT
        cv.claim_number,
        MAX(cv.claim_version_id) AS claim_version_id,
        cv.claim_id,
        cv.policy_number,
        MAX(cv.event_enqueued_utc_time) AS latest_event_time
    FROM prod_adp_certified.claim.claim_version cv
    WHERE DATE(cv.event_enqueued_utc_time) = '{date_str}'
    GROUP BY cv.claim_number, cv.claim_id, cv.policy_number
    """
    return spark.sql(query)


def get_claim_details(spark, date_str):
    """Get detailed claim information for specified date"""
    query = f"""
    WITH latest_claims AS (
        SELECT DISTINCT
            cv.claim_number,
            MAX(cv.claim_version_id) AS claim_version_id,
            cv.claim_id,
            cv.policy_number,
            MAX(cv.event_enqueued_utc_time) AS latest_event_time
        FROM prod_adp_certified.claim.claim_version cv
        WHERE DATE(cv.event_enqueued_utc_time) = '{date_str}'
        GROUP BY cv.claim_number, cv.claim_id, cv.policy_number
    )
    SELECT DISTINCT 
        cv.claim_number,
        cv.policy_number,
        cv.claim_version_id,
        cvi.claim_version_item_index,
        cv.policy_cover_type,
        cvi.claim_item_type,
        cvi.vehicle_unattended,
        cvi.excesses_applied,
        cvi.total_loss_flag,
        cl.is_first_party,
        i.start_date,
        i.reported_date,
        i.multiple_parties_involved,
        i.notification_method,
        i.impact_speed,
        i.impact_speed_unit,
        i.impact_speed_range,
        hour(i.start_date) as incidentHourC,
        dayofweek(i.start_date) as incidentDayOfWeekC,
        month(i.start_date) as incidentMonthC,
        i.incident_type,
        i.incident_cause,
        i.incident_sub_cause,
        i.circumstances,
        v.year_of_manufacture as manufacture_yr_claim,
        v.outstanding_finance_amount,
        c.first_party_confirmed_tp_notified_claim,
        cv.claim_id,
        es.is_air_ambulance_attendance,
        es.is_ambulance_attendance,
        es.is_crime_reference_provided,
        es.is_fire_service_attendance,
        es.is_police_attendance,
        es.police_considering_actions,
        dd.assessment_category,
        dd.boot_opens,
        dd.doors_open,
        dd.engine_damage,
        dd.front_severity,
        dd.front_bonnet_severity,
        dd.front_left_severity,
        dd.front_right_severity,
        dd.left_severity,
        dd.left_back_seat_severity,
        dd.left_front_wheel_severity,
        dd.left_mirror_severity,
        dd.left_rear_wheel_severity,
        dd.left_underside_severity,
        dd.rear_severity,
        dd.rear_left_severity,
        dd.rear_right_severity,
        dd.rear_window_damage_severity,
        dd.right_severity,
        dd.right_back_seat_severity,
        dd.right_front_wheel_severity,
        dd.right_mirror_severity,
        dd.right_rear_wheel_severity,
        dd.right_roof_severity,
        dd.right_underside_severity,
        dd.roof_damage_severity,
        dd.underbody_damage_severity,
        dd.windscreen_damage_severity,
        lc.latest_event_time,
        d.date_of_birth as claim_driver_dob
    FROM latest_claims lc
    INNER JOIN prod_adp_certified.claim.claim_version cv
        ON lc.claim_version_id = cv.claim_version_id
    INNER JOIN prod_adp_certified.claim.claim_version_item cvi
        ON cv.claim_version_id = cvi.claim_version_id
        AND lc.claim_id = cvi.claim_id
    INNER JOIN prod_adp_certified.claim.claim c
        ON c.claim_id = cv.claim_id
    LEFT JOIN prod_adp_certified.claim.damage_details dd
        ON dd.event_identity = cv.event_identity
        AND dd.claim_version_item_index = cvi.claim_version_item_index
    LEFT JOIN prod_adp_certified.claim.incident i
        ON cv.event_identity = i.event_identity
    LEFT JOIN prod_adp_certified.claim.vehicle v
        ON cv.event_identity = v.event_identity
        AND cvi.claim_version_item_index = v.claim_version_item_index
    LEFT JOIN prod_adp_certified.claim.claimant cl
        ON cl.claim_version_id = cvi.claim_version_id
        AND cl.claim_version_item_index = cvi.claim_version_item_index
    LEFT JOIN prod_adp_certified.claim.emergency_services es
        ON cv.event_identity = es.event_identity
    LEFT JOIN prod_adp_certified.claim.driver d
        ON d.claim_version_item_index = cvi.claim_version_item_index
        AND d.event_identity = cvi.event_identity
    WHERE cl.is_first_party = true
        AND cvi.claim_item_type = 'CarMotorVehicleClaimItem'
        AND cvi.claim_version_item_index = 0
    """
    return spark.sql(query)


def apply_damage_calculations(df, spark):
    """Apply damage score calculations to claims data"""
    # Register UDF
    calculate_damage_score_udf = udf(calculate_damage_score, StructType([
        StructField("damageScore", IntegerType(), False),
        StructField("areasDamagedMinimal", IntegerType(), False),
        StructField("areasDamagedMedium", IntegerType(), False),
        StructField("areasDamagedHeavy", IntegerType(), False),
        StructField("areasDamagedSevere", IntegerType(), False)
    ]))
    
    # List of damage columns
    damage_columns = [
        'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 
        'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
        'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 
        'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
        'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
        'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity'
    ]
    
    # Apply UDF
    df = df.withColumn(
        "damage_scores",
        calculate_damage_score_udf(*[df[col] for col in damage_columns])
    )
    
    # Split struct columns
    df = df.select(
        "*",
        "damage_scores.damageScore",
        "damage_scores.areasDamagedMinimal",
        "damage_scores.areasDamagedMedium",
        "damage_scores.areasDamagedHeavy",
        "damage_scores.areasDamagedSevere"
    ).withColumn(
        "areasDamagedTotal", 
        col("areasDamagedMinimal") + col("areasDamagedMedium") + col("areasDamagedSevere") + col("areasDamagedHeavy")
    ).withColumn(
        "veh_age", 
        round(datediff(col("start_date"), to_date(concat(col("manufacture_yr_claim"), lit('-01-01')))) / 365.25, 0)
    ).withColumn(
        "veh_age_more_than_10", 
        (col("veh_age") > 10).cast("int")
    ).withColumn(
        "claim_driver_age",
        round(datediff(col("start_date"), to_date(col("claim_driver_dob"))) / 365.25)
    ).drop("damage_scores")
    
    return df


def dedup_driver_features(df):
    """Deduplicate driver features by getting minimum age"""
    min_drv_age = df.groupBy("claim_number").agg(
        min(col("claim_driver_age")).alias("min_claim_driver_age")
    )
    
    df = df.drop("claim_driver_age").join(
        min_drv_age, on="claim_number", how="left"
    ).drop("claim_driver_dob").dropDuplicates()
    
    return df


def get_policy_features(spark):
    """Get policy features for claims"""
    query = """
    SELECT DISTINCT
        pt.policy_number,
        pt.policy_transaction_id,
        pt.policy_start_date,
        pt.policy_renewal_date,
        pt.policy_type,
        qi.session_id as quote_session_id,
        qi.quote_iteration_id,
        v.vehicle_use_code as vehicle_use_quote,
        pcp.policyholder_ncd_years,
        pcp.ncd_protected_flag,
        pt.sales_channel,
        pv.overnight_location_abi_code,
        pv.vehicle_overnight_location_id,
        pv.vehicle_overnight_location_name,
        pv.business_mileage,
        pv.annual_mileage,
        v.year_of_manufacture,
        v.registration_date,
        v.car_group,
        v.vehicle_value,
        v.purchase_date,
        cpv.voluntary_amount
    FROM prod_adp_certified.policy_motor.policy_transaction pt
    LEFT JOIN prod_adp_certified.quote_motor.quote_iteration qi
        ON pt.quote_iteration_id = qi.quote_iteration_id
    LEFT JOIN prod_adp_certified.quote_motor.vehicle v
        ON qi.quote_iteration_id = v.quote_iteration_id
    LEFT JOIN prod_adp_certified.policy_motor.vehicle pv
        ON pt.policy_transaction_id = pv.policy_transaction_id
    LEFT JOIN prod_adp_certified.policy_motor.policy_car_products pcp
        ON pt.policy_transaction_id = pcp.policy_transaction_id
    LEFT JOIN prod_adp_certified.policy_motor.comprehensive_policy_vehicle cpv
        ON pt.policy_transaction_id = cpv.policy_transaction_id
    """
    return spark.sql(query)


def aggregate_driver_columns(policy_df):
    """Aggregate driver-level features across multiple drivers"""
    driver_cols = [
        'additional_vehicles_owned', 'age_at_policy_start_date', 
        'cars_in_household', 'licence_length_years', 'years_resident_in_uk'
    ]
    
    for col_name in driver_cols:
        policy_df = policy_df.withColumn(
            f"max_{col_name}", 
            greatest(*[col(f"{col_name}_{i}") for i in range(1, 6)])
        )
        policy_df = policy_df.withColumn(
            f"min_{col_name}", 
            least(*[col(f"{col_name}_{i}") for i in range(1, 6)])
        )
    
    # Drop individual driver columns except driver 1
    drop_cols = []
    for col_name in driver_cols:
        for i in range(2, 6):
            drop_cols.append(f"{col_name}_{i}")
    
    policy_df = policy_df.drop(*drop_cols)
    
    return policy_df
</content>