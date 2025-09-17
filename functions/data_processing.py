import os
import random
import pandas as pd
import math
from math import floor, ceil
import re
from typing import Tuple, List, Dict
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, row_number, greatest, least, collect_list, lower, mean, when, regexp_replace, min, max, datediff, to_date, concat, lit, round, date_format, hour, udf, mode
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql import Window
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pyspark.sql import functions as F
from pathlib import Path


class DataPreprocessing:
    """
    Core data preprocessing logic for SVI (Single Vehicle Incident) checks.
    Handles data loading, filtering, cleaning, and train/test splitting.
    """
    
    def __init__(self, spark, env_config, start_date="2023-01-01"):
        self.spark = spark
        self.start_date = start_date
        self.env_config = env_config
        self.calculate_damage_score_udf = self.register_udf()

    def get_table_path(self, schema, table, catalog_type='auxiliary'):
        """
        Constructs a fully qualified table path based on the current environment.
        
        Parameters:
        schema: The schema name (e.g., 'single_vehicle_incident_checks')
        table: The table name (e.g., 'claim_referral_log')
        catalog_type: Either 'auxiliary' (for reading), 'mlstore' (for writing), or 'adp' (for ADP certified data)
        
        Returns:
        Fully qualified table path (e.g., 'prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claim_referral_log')
        """
        
        if catalog_type == 'auxiliary':
            catalog = self.env_config['auxiliary_catalog']
        elif catalog_type == 'mlstore':
            catalog = self.env_config['mlstore_catalog']
        elif catalog_type == 'adp':
            catalog = self.env_config['adp_catalog']
        else:
            raise ValueError(f"Invalid catalog_type: {catalog_type}. Must be 'auxiliary', 'mlstore', or 'adp'")
        
        return f"{catalog}.{schema}.{table}"

    # Get the current working directory and project root
    try:
        current_notebook_path = Path(os.getcwd())
        
        # Find project root by looking for common project markers
        # This makes the config discovery more robust
        project_root = None
        search_path = current_notebook_path
        
        # Search upward for project root indicators
        while search_path.parent != search_path:
            if (search_path / "configs").exists() and (search_path / "notebooks").exists():
                project_root = search_path
                break
            if (search_path / "bundle.yaml").exists() or (search_path / "requirements.txt").exists():
                project_root = search_path
                break
            search_path = search_path.parent
        
        if project_root is None:
            # Fallback: assume we're in notebooks/ and project root is parent
            if current_notebook_path.name == "notebooks" or "notebooks" in current_notebook_path.parts:
                idx = list(current_notebook_path.parts).index("notebooks") if "notebooks" in current_notebook_path.parts else -1
                if idx >= 0:
                    project_root = Path(*current_notebook_path.parts[:idx])
                else:
                    project_root = current_notebook_path.parent
            else:
                project_root = current_notebook_path
                
        print(f"Project root detected: {project_root}")
        print(f"Current location: {current_notebook_path}")
        
    except Exception as e:
        print(f"Error detecting project structure: {e}")
        current_notebook_path = Path(".")
        project_root = Path(".")
        
    def load_claim_referral_log(self):
        """
        Load and process claim referral log data from auxiliary data.
        """
        # Get dynamic table path
        table_path = self.get_table_path("single_vehicle_incident_checks", "claim_referral_log", "auxiliary")
        
        print(f"Loading claim referral log from: {table_path}")
        
        clm_log_df = self.spark.sql(f"""
            SELECT DISTINCT * 
            FROM {table_path}
            WHERE `Date_received` >= '{self.start_date}'
        """)
        
        # Process claim referral log
        clm_log_df = clm_log_df.withColumn("Claim_Ref", regexp_replace("Claim_Ref", "\\*", ""))
        
        # Create risk indicators
        risk_cols = {
            "Final_Outcome_of_Claim": ["Withdrawn whilst investigation ongoing", 
                                     "Repudiated - Litigated - Claim then discontinued", 
                                     "Repudiated - Litigated - Success at trial", 
                                     "Repudiated - Not challenged"],
            "Outcome_of_Referral": ["Accepted"],
            "Outcome_of_Investigation": ["Repudiated", "Repudiated in part", 
                                       "Under Investigation", "Withdrawn whilst investigation ongoing"]
        }
        
        for this_col in risk_cols:
            clm_log_df = clm_log_df.withColumn(
                f'{this_col}_risk', 
                col(this_col).isin(*risk_cols[this_col]).cast('integer')
            )
        
        clm_log_df = clm_log_df.fillna({
            "Final_Outcome_of_Claim_risk": 0, 
            "Outcome_of_Referral_risk": 0, 
            "Outcome_of_Investigation_risk": 0
        })
        
        clm_log_df = clm_log_df.withColumn(
            "fraud_risk", 
            greatest("Final_Outcome_of_Claim_risk", "Outcome_of_Referral_risk", 
                    "Outcome_of_Investigation_risk")
        )
        
        # Filter for FC claims and select relevant columns
        clm_log = clm_log_df.filter(
            lower(col("Claim_Ref")).contains("fc/")
        ).select(
            col("Claim_Ref").alias("id"), 
            "fraud_risk",
            "Outcome_of_Investigation"
        )
        
        print(f"Loaded {clm_log.count()} claim referral records")
        return clm_log
    
    def load_svi_performance_data(self, clm_log):
        """
        Load SVI performance data and create target variable.
        """
        # Get dynamic table path
        table_path = self.get_table_path("single_vehicle_incident_checks", "svi_performance", "auxiliary")
        
        print(f"Loading SVI performance data from: {table_path}")
        
        clm_log.createOrReplaceTempView("clm_log")
        
        df = self.spark.sql(f"""
            SELECT DISTINCT
                svi.`Claim_Number` as claim_number, 
                svi.`Result_of_Outsourcing` as TBG_Outcome, 
                svi.`FA_Outcome` as FA_Outcome,
                log.fraud_risk,
                log.Outcome_of_Investigation as Outcome_of_Investigation,
                CASE 
                    WHEN lower(svi.`Result_of_Outsourcing`) = 'settled' THEN 0 
                    WHEN lower(svi.`Result_of_Outsourcing`) IN ('withdrawn', 'repudiated', 'managed away', 'cancelled') THEN 1
                END AS tbg_risk,
                CASE 
                    WHEN lower(svi.`FA_Outcome`) IN ('claim closed', "claim to review", 'not comprehensive cover') THEN 1 
                    ELSE 0 
                END AS fa_risk
            FROM {table_path} svi
            LEFT JOIN clm_log log 
            ON lower(svi.`Claim_Number`) = lower(log.id)
            WHERE svi.`Notification_Date` >= '{self.start_date}'
            AND (lower(svi.`Result_of_Outsourcing`) IS NULL OR lower(svi.`Result_of_Outsourcing`) NOT IN ('ongoing - client', 'ongoing - tbg', 'pending closure'))
            AND lower(svi.`FA_Outcome`) != 'not comprehensive cover'
        """)
        
        # Create target variable: claim is high risk if flagged at either stage
        target_df = df.withColumn(
            "svi_risk", 
            greatest(col("fraud_risk"), col("tbg_risk"))
        ).fillna({"svi_risk": -1})
        
        print(f"Created target variable for {target_df.count()} claims")
        return target_df

    def get_latest_claim_version(self, target_df):
        """
        Get the latest claim version for each claim.
        """
        # Set catalog to ADP certified
        self.spark.sql(f'USE CATALOG {self.env_config["adp_catalog"]}')
        
        target_df.createOrReplaceTempView("target_df")
        
        latest_claim_version = self.spark.sql("""
            SELECT DISTINCT
                MAX(cv.claim_number) AS claim_number,
                MAX(svi.svi_risk) AS svi_risk, 
                MAX(svi.tbg_risk) AS tbg_risk, 
                MAX(svi.FA_Outcome) AS FA_Outcome, 
                MAX(svi.fa_risk) AS fa_risk,
                MAX(svi.fraud_risk) AS fraud_risk, 
                MAX(svi.Outcome_of_Investigation) AS Outcome_of_Investigation,
                MAX(cv.claim_version_id) AS claim_version_id,
                cv.claim_id,
                MAX(cv.event_enqueued_utc_time) AS latest_event_time
            FROM target_df svi
            LEFT JOIN claim.claim_version cv
            ON LOWER(cv.claim_number) = LOWER(svi.claim_number)
            GROUP BY cv.claim_id
            HAVING claim_number IS NOT NULL
        """)
        
        return latest_claim_version

    def load_claim_data(self, latest_claim_version):
        """
        Load and join claim data from multiple ADP tables.
        """

        latest_claim_version.createOrReplaceTempView("latest_claim_version")
        
        check_df = self.spark.sql("""
            SELECT DISTINCT 
                claim.claim_version.claim_number,
                claim.claim_version.policy_number, 
                claim.claim_version.claim_version_id,
                claim.claim_version_item.claim_version_item_index, 
                claim.claim_version.policy_cover_type,
                claim.claim_version.position_status,
                claim.claim_version_item.claim_item_type, 
                claim.claim_version_item.not_on_mid, 
                claim.claim_version_item.vehicle_unattended,
                claim.claim_version_item.excesses_applied,
                claim.claim_version_item.total_loss_date, 
                claim.claim_version_item.total_loss_flag,
                claim.claim_version_item.first_party as cvi_first_party,
                claim.claimant.is_first_party,
                incident.event_identity as incident_event_identity,
                lcv.latest_event_time,
                claim.incident.start_date,
                claim.incident.reported_date,
                claim.incident.multiple_parties_involved,
                claim.incident.notification_method,
                claim.incident.impact_speed,
                claim.incident.impact_speed_unit,
                claim.incident.impact_speed_range,
                hour(claim.incident.start_date) as incidentHourC,
                dayofweek(claim.incident.start_date) as incidentDayOfWeekC,
                month(claim.incident.start_date) as incidentMonthC,
                claim.incident.incident_location_longitude,
                claim.incident.incident_type,
                claim.incident.incident_cause,
                claim.incident.incident_sub_cause,
                claim.incident.circumstances, 
                claim.vehicle.year_of_manufacture as manufacture_yr_claim,
                claim.vehicle.outstanding_finance_amount,
                claim.driver.driver_id,
                claim.driver.date_of_birth as claim_driver_dob,
                YEAR(claim.incident.start_date) - YEAR(claim.driver.date_of_birth) as claim_driver_age,
                claim.claim.first_party_confirmed_tp_notified_claim,
                claim.claim_version.claim_id,
                claim.emergency_services.is_air_ambulance_attendance, 
                claim.emergency_services.is_ambulance_attendance, 
                claim.emergency_services.is_crime_reference_provided, 
                claim.emergency_services.is_fire_service_attendance, 
                claim.emergency_services.is_police_attendance,  
                claim.emergency_services.police_considering_actions, 
                claim.damage_details.assessment_category,
                claim.damage_details.boot_opens,
                claim.damage_details.doors_open,
                claim.damage_details.engine_damage,
                claim.damage_details.front_severity, 
                claim.damage_details.front_bonnet_severity,
                claim.damage_details.front_left_severity,
                claim.damage_details.front_right_severity,
                claim.damage_details.left_severity,
                claim.damage_details.left_back_seat_severity,
                claim.damage_details.left_front_wheel_severity,
                claim.damage_details.left_mirror_severity,
                claim.damage_details.left_rear_wheel_severity,
                claim.damage_details.left_underside_severity,
                claim.damage_details.rear_severity,
                claim.damage_details.rear_left_severity,
                claim.damage_details.rear_right_severity,
                claim.damage_details.rear_window_damage_severity,
                claim.damage_details.right_severity,
                claim.damage_details.right_back_seat_severity,
                claim.damage_details.right_front_wheel_severity,
                claim.damage_details.right_mirror_severity,
                claim.damage_details.right_rear_wheel_severity,
                claim.damage_details.right_roof_severity,
                claim.damage_details.right_underside_severity,
                claim.damage_details.roof_damage_severity,
                claim.damage_details.underbody_damage_severity,
                claim.damage_details.windscreen_damage_severity,
                lcv.tbg_risk, 
                lcv.fraud_risk, 
                lcv.Outcome_of_Investigation,
                lcv.svi_risk, 
                lcv.FA_Outcome, 
                lcv.fa_risk
            FROM latest_claim_version lcv
            INNER JOIN claim.claim_version
                ON lcv.claim_number = claim_version.claim_number 
            INNER JOIN claim.claim_version_item
                ON lcv.claim_version_id = claim_version.claim_version_id
                AND claim_version.claim_version_id = claim_version_item.claim_version_id
                AND lcv.claim_id = claim_version_item.claim_id
            INNER JOIN claim.claim
                ON claim.claim_id = claim_version.claim_id
                AND claim.claim_id = claim_version_item.claim_id
            LEFT JOIN claim.damage_details
                ON damage_details.event_identity = claim_version.event_identity
                AND damage_details.claim_version_item_index = claim_version_item.claim_version_item_index
            LEFT JOIN claim.incident
                ON claim_version.event_identity = incident.event_identity
            LEFT JOIN claim.vehicle
                ON claim_version.event_identity = vehicle.event_identity
                AND claim_version_item.claim_version_item_index = vehicle.claim_version_item_index
            LEFT JOIN claim.claimant
                ON claimant.claim_version_id = claim_version_item.claim_version_id
                AND claimant.claim_version_item_index = claim_version_item.claim_version_item_index
                AND claimant.event_identity = claim_version_item.event_identity
            LEFT JOIN claim.emergency_services
                ON claim.claim_version.event_identity = emergency_services.event_identity
            LEFT JOIN claim.driver
                ON claim.driver.claim_version_item_index = claim_version_item.claim_version_item_index
                AND claim.driver.event_identity = claim_version_item.event_identity
                AND claim_version.event_identity = claim.driver.event_identity
            WHERE claim_version.claim_number IS NOT NULL
            AND claim.claimant.is_first_party = true
            AND claim_version_item.claim_item_type='CarMotorVehicleClaimItem'
            AND claim_version_item.claim_version_item_index=0
            AND year(incident.start_date)>=2023
        """)
        
        # Fix data types for check variables
        check_df = check_df.withColumn(
                                    'police_considering_actions', 
                                    when(col('police_considering_actions').isNull(), False)
                                    .otherwise(col('police_considering_actions').cast('boolean')))
        check_df = check_df.withColumn(
                                    'is_crime_reference_provided', 
                                    when(col('is_crime_reference_provided').isNull(), False)
                                    .otherwise(col('is_crime_reference_provided').cast('boolean'))
                                )
        check_df = check_df.withColumn(
                                       'multiple_parties_involved', 
                                    when(col('multiple_parties_involved').isNull(), False)
                                    .otherwise(col('multiple_parties_involved').cast('boolean')))
        
        print(f"Loaded claim data for {check_df.count()} claims")
        return check_df


    def load_policy_data(self, df):
        """
        Load policy data from auxiliary data.
        """

        start_day = df.select(min("reported_date")).collect()[0][0]
        end_day = df.select(max("reported_date")).collect()[0][0]

        w = Window.partitionBy("policy_number").orderBy(F.col("transaction_timestamp").desc())

        policy_transaction = self.spark.sql("""
                            SELECT 
                                -- Columns from policy_transaction table
                                pt.policy_transaction_id,
                                pt.sales_channel, 
                                pt.transaction_timestamp,
                                pt.quote_session_id,
                                pt.policy_number
                                FROM prod_adp_certified.policy.policy_transaction pt """
                            ).withColumn("rn", F.row_number().over(w))\
                            .filter(F.col("rn") == 1)\
                            .drop("rn", "transaction_timestamp")

        policy = self.spark.sql(""" 
                            SELECT
                            p.policy_number,
                            p.policy_start_date,
                            p.policy_renewal_date,
                            p.policy_type,
                            p.policyholder_ncd_years,
                            p.ncd_protected_flag,
                            p.policy_number FROM prod_adp_certified.policy.policy p
                            """)

        vehicle = self.spark.sql(""" 
                            SELECT 
                            v.policy_transaction_id,
                            v.overnight_location_abi_code,
                            vo.vehicle_overnight_location_id, 
                            vo.vehicle_overnight_location_name, 
                            v.business_mileage, 
                            v.annual_mileage, 
                            v.year_of_manufacture, 
                            v.registration_date, 
                            v.car_group, 
                            v.vehicle_value, 
                            v.vehicle_registration,
                            v.purchase_date FROM prod_adp_certified.policy.vehicle v 
                            LEFT JOIN prod_adp_certified.reference_motor.vehicle_overnight_location vo 
                            ON v.overnight_location_abi_code = vo.vehicle_overnight_location_code
                            """)

        excess = self.spark.sql(""" 
                        SELECT 
                        policy_transaction_id,
                        voluntary_amount
                        from prod_adp_certified.policy.excess 
                        WHERE excess_index = 0
                        """)

        driver = self.spark.sql(""" select
                        pd.policy_transaction_id,
                        pd.date_of_birth,
                        pd.additional_vehicles_owned, 
                        pd.age_at_policy_start_date, 
                        pd.cars_in_household, 
                        pd.licence_length_years, 
                        pd.years_resident_in_uk,
                        do.employment_type_abi_code
                        from prod_adp_certified.policy.driver pd
                        LEFT JOIN prod_adp_certified.policy.driver_occupation do
                        ON pd.policy_transaction_id = do.policy_transaction_id
                        AND pd.driver_index = do.driver_index
                        WHERE do.occupation_index = 1
                        ORDER BY pd.policy_transaction_id,pd.driver_index
            """).dropDuplicates()
        
        driver_transformed = driver.groupBy("policy_transaction_id").agg(
            F.collect_list("date_of_birth").alias("date_of_birth"),
            F.collect_list("additional_vehicles_owned").alias("additional_vehicles_owned"),
            F.collect_list("age_at_policy_start_date").alias("age_at_policy_start_date"),
            F.collect_list("cars_in_household").alias("cars_in_household"),
            F.collect_list("licence_length_years").alias("licence_length_years"),
            F.collect_list("years_resident_in_uk").alias("years_resident_in_uk"),
            F.collect_list("employment_type_abi_code").alias("employment_type_abi_code")
        )
        
        
        max_list_size = driver_transformed.select(
            *[F.size(F.col(col)).alias(col) for col in driver_transformed.columns if col != "policy_transaction_id"]
        ).agg(F.max(F.greatest(*[F.col(col) for col in driver_transformed.columns if col != "policy_transaction_id"]))).collect()[0][0]
        
        # Dynamically explode each list into individual columns
        columns_to_explode = [col for col in driver_transformed.columns if col != "policy_transaction_id"]
        for col in columns_to_explode:
            for i in range(max_list_size):
                driver_transformed = driver_transformed.withColumn(
                    f"{col}_{i+1}",
                    F.col(col)[i]
                )
        # Drop the original list columns
        driver_transformed = driver_transformed.drop(*columns_to_explode)

        svi_claims = self.spark.sql("""
                            SELECT DISTINCT
                                c.claim_id,
                                c.policy_number,
                                i.reported_date
                            FROM
                            prod_adp_certified.claim.incident i
                            INNER JOIN
                                prod_adp_certified.claim.claim_version cv
                            ON i.event_identity = cv.event_identity
                            INNER JOIN
                                prod_adp_certified.claim.claim c
                                ON cv.claim_id = c.claim_id
                            WHERE
                            incident_cause IN 
                            ('Animal', 
                            'Attempted To Avoid Collision', 
                            'Debris/Object', 
                            'Immobile Object', 
                            'Lost Control - No Third Party Involved')
        """).filter(f"DATE(reported_date)>='{start_day}' AND DATE(reported_date)<='{end_day}'")
 

        policy_svi = (
            svi_claims
            .join(policy, ['policy_number'], "left")
            .join(policy_transaction, ['policy_number'], "left")
            .join(vehicle, ['policy_transaction_id'], "left")
            .join(excess, ['policy_transaction_id'], "left")
            .join(driver_transformed, ['policy_transaction_id'], "left")
            .drop_duplicates()
        ).withColumn("postcode", F.lit(None)) #ignore postcode

        pol_cols = [
            'policy_transaction_id', 'policy_number', 'quote_session_id', 
            'policy_start_date', 'policy_renewal_date', 'policy_type', 
            'policyholder_ncd_years', 'ncd_protected_flag', 'sales_channel', 
            'overnight_location_abi_code', 'vehicle_overnight_location_id', 
            'vehicle_overnight_location_name', 'business_mileage', 'annual_mileage', 
            'year_of_manufacture', 'registration_date', 'car_group', 'vehicle_value', 
            'purchase_date', 'voluntary_amount', 'date_of_birth_1', 
            'additional_vehicles_owned_1', 'additional_vehicles_owned_2', 
            'additional_vehicles_owned_3', 'additional_vehicles_owned_4', 
            'additional_vehicles_owned_5', 'age_at_policy_start_date_1', 
            'age_at_policy_start_date_2', 'age_at_policy_start_date_3', 
            'age_at_policy_start_date_4', 'age_at_policy_start_date_5', 
            'cars_in_household_1', 'cars_in_household_2', 'cars_in_household_3', 
            'cars_in_household_4', 'cars_in_household_5', 'licence_length_years_1', 
            'licence_length_years_2', 'licence_length_years_3', 'licence_length_years_4', 
            'licence_length_years_5', 'years_resident_in_uk_1', 'years_resident_in_uk_2', 
            'years_resident_in_uk_3', 'years_resident_in_uk_4', 'years_resident_in_uk_5', 
            'employment_type_abi_code_1', 'employment_type_abi_code_2', 
            'employment_type_abi_code_3', 'employment_type_abi_code_4', 
            'employment_type_abi_code_5', 'postcode'
        ]

        policy_svi = policy_svi.select(pol_cols)

        # Add vehicle use from quotes
        quote_iteration_table = f"{self.env_config['adp_catalog']}.quote_motor.quote_iteration"
        vehicle_table = f"{self.env_config['adp_catalog']}.quote_motor.vehicle"

        quote_iteration_df = self.spark.table(quote_iteration_table)
        vehicle_df = self.spark.table(vehicle_table)

        policy_svi = policy_svi.join(
            quote_iteration_df, 
            policy_svi.quote_session_id == quote_iteration_df.session_id, 
            "left"
        ).join(
            vehicle_df, 
            "quote_iteration_id", 
            "left"
        ).select(
            policy_svi["*"],
            quote_iteration_df.session_id,
            vehicle_df.vehicle_use_code.alias("vehicle_use_quote"),
            quote_iteration_df.quote_iteration_id
        )

        # Get latest transaction per policy
        window_spec = Window.partitionBy(F.col("policy_number")).orderBy(F.col("policy_transaction_id").desc())
        policy_svi = policy_svi.withColumn(
            "row_num", 
            row_number().over(window_spec)
        ).filter(F.col("row_num") == 1).drop("row_num")

        # Aggregate driver columns
        driver_cols = [
            'additional_vehicles_owned', 'age_at_policy_start_date', 
            'cars_in_household', 'licence_length_years', 'years_resident_in_uk'
        ]

        for col_name in driver_cols:
            policy_svi = policy_svi.withColumn(
                f"max_{col_name}", 
                greatest(*[F.col(f"{col_name}_{i}") for i in range(1, 6)])
            ).withColumn(
                f"min_{col_name}", 
                least(*[F.col(f"{col_name}_{i}") for i in range(1, 6)])
            )

        # Drop redundant columns
        drop_cols = []
        for col_base in driver_cols:
            drop_cols.extend([f"{col_base}_{i}" for i in range(2, 6)])

        policy_svi = policy_svi.drop(*drop_cols)

        print(f"Loaded policy data for {policy_svi.count()} policies")
        return policy_svi
    
    def deduplicate_driver_data(self, check_df):
        """
        Deduplicate driver data by taking minimum driver age per claim.
        """
        # Get minimum claim driver age for each claim
        min_drv_age = check_df.groupBy("claim_number").agg(
            min(col("claim_driver_age")).alias("min_claim_driver_age")
        )
        
        # Join back and drop duplicates
        check_df = check_df.drop("claim_driver_age").join(
            min_drv_age, 
            on="claim_number", 
            how="left"
        ).drop("driver_id", "claim_driver_dob").dropDuplicates()
        
        return check_df
    
    def join_claim_and_policy_data(self, check_df, policy_svi):
        """
        Join claim and policy data, filter for matched policies only.
        """
        check_df = check_df.join(policy_svi, on="policy_number", how="left")
        
        # Filter for claims with matched policies only
        check_df = check_df.filter(col("policy_transaction_id").isNotNull()).dropDuplicates()
        
        print(f"Joined claim and policy data: {check_df.count()} records")
        return check_df

    def register_udf(self):
        """
        Register the UDF for calculating damage scores.
        """
        def calculate_damage_score(*args):
            """
            UDF to calculate damage score based on severity levels.
            
            Scoring:
            - Base score starts at 1
            - 2x for Minimal damage
            - 3x for Medium damage
            - 4x for Heavy damage
            - 5x for Severe damage
            
            Returns:
                tuple: (damageScore, areasDamagedMinimal, areasDamagedMedium, areasDamagedHeavy, areasDamagedSevere)
            """
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

        # Register UDF
        return udf(calculate_damage_score, StructType([
            StructField("damageScore", IntegerType(), False),
            StructField("areasDamagedMinimal", IntegerType(), False),
            StructField("areasDamagedMedium", IntegerType(), False),
            StructField("areasDamagedHeavy", IntegerType(), False),
            StructField("areasDamagedSevere", IntegerType(), False)
        ]))
    
    def calculate_damage_scores(self, check_df):
        """
        Calculate damage scores and aggregate damage areas by severity.
        """
        # List of damage columns
        damage_columns = [
            'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 
            'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
            'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 
            'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
            'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
            'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity'
        ]
        
        # Filter to only existing columns
        existing_damage_columns = [col for col in damage_columns if col in check_df.columns]
        
        if not existing_damage_columns:
            print("No damage severity columns found in dataframe")
            # Add empty damage columns if none exist
            check_df = check_df.withColumn("damageScore", lit(0))
            check_df = check_df.withColumn("areasDamagedMinimal", lit(0))
            check_df = check_df.withColumn("areasDamagedMedium", lit(0))
            check_df = check_df.withColumn("areasDamagedHeavy", lit(0))
            check_df = check_df.withColumn("areasDamagedSevere", lit(0))
            check_df = check_df.withColumn("areasDamagedTotal", lit(0))
            return check_df
        
        # Apply the UDF to calculate damage scores
        check_df = check_df.withColumn(
            "damage_scores",
            self.calculate_damage_score_udf(*[check_df[col] for col in existing_damage_columns])
        )
        
        # Split the struct column into separate columns
        check_df = check_df.select(
            "*",
            "damage_scores.damageScore",
            "damage_scores.areasDamagedMinimal",
            "damage_scores.areasDamagedMedium",
            "damage_scores.areasDamagedHeavy",
            "damage_scores.areasDamagedSevere"
        ).withColumn(
            "areasDamagedTotal", 
            col("areasDamagedMinimal") + col("areasDamagedMedium") + 
            col("areasDamagedSevere") + col("areasDamagedHeavy")
        ).drop("damage_scores")
        
        print("Damage score calculations completed")
        return check_df
    
    def calculate_vehicle_and_driver_features(self, check_df):
        """
        Calculate vehicle age and driver age features.
        """
        # Vehicle age calculation
        check_df = check_df.withColumn(
            "veh_age", 
            round(datediff(col("start_date"), to_date(concat(col("manufacture_yr_claim"), lit('-01-01')))) / 365.25, 0)
        )
        
        # Vehicle age flag
        check_df = check_df.withColumn(
            "veh_age_more_than_10", 
            (col("veh_age") > 10).cast("int")
        )
        
        # Driver age flags
        check_df = check_df.withColumn(
            "driver_age_low_1", 
            when(col("age_at_policy_start_date_1") < 25, 1)
            .when(col("age_at_policy_start_date_1").isNull(), 1)
            .otherwise(0)
        )
        
        check_df = check_df.withColumn(
            "claim_driver_age_low", 
            when(col("min_claim_driver_age") < 25, 1)
            .when(col("min_claim_driver_age").isNull(), 1)
            .otherwise(0)
        )
        
        # Licence length flag
        check_df = check_df.withColumn(
            "licence_low_1", 
            when(col("licence_length_years_1") <= 3, 1).otherwise(0)
        )
        
        print("Vehicle and driver features calculated")
        return check_df
    
    def create_check_variables(self, check_df):
        """
        Create check variables (C1-C14) for fraud risk assessment.
        """
        # C1: Friday/Saturday night incidents
        check_df = check_df.withColumn("incident_day_of_week", date_format(col("latest_event_time"), "E"))
        
        fri_sat_night = (
            (col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | 
            (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4)))
        )
        
        check_df = check_df.withColumn(
            "C1_fri_sat_night",
            when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0)
        )
        
        check_df = check_df.withColumn("reported_day_of_week", date_format(col("latest_event_time"), "E"))
        
        # C2: Reporting delay (3+ days)
        check_df = check_df.withColumn("delay_in_reporting", datediff(col("reported_date"), col("start_date")))
        check_df = check_df.withColumn(
            "C2_reporting_delay", 
            when(col("delay_in_reporting") >= 3, 1)
            .when(col("delay_in_reporting").isNull(), 1)
            .otherwise(0)
        )
        
        # Weekend and Monday reporting flags
        check_df = check_df.withColumn(
            "is_incident_weekend",
            when(date_format(col("start_date"), "E").isin("Fri", "Sat", "Sun"), 1).otherwise(0)
        )
        
        check_df = check_df.withColumn(
            "is_reported_monday",
            when(date_format(col("reported_date"), "E") == "Mon", 1).otherwise(0)
        )
        
        # C3: Weekend incident reported on Monday
        check_df = check_df.withColumn(
            "C3_weekend_incident_reported_monday",
            when((col("is_incident_weekend") == 1) & (col("is_reported_monday") == 1), 1).otherwise(0)
        )
        
        # C5: Night incident (11pm - 5am)
        check_df = check_df.withColumn(
            "C5_is_night_incident",
            when((hour(col("start_date")) >= 23) | (hour(col("start_date")) <= 5) | (hour(col("start_date"))).isNull(), 1)
            .otherwise(0)
        )
        
        # C6: No commuting policy but rush hour incident
        not_commuting_rush = (
            (lower(col("vehicle_use_quote")) == "1") & 
            ((hour(col("start_date")).between(6, 10)) | (hour(col("start_date")).between(15, 18)))
        )
        check_df = check_df.withColumn(
            "C6_no_commuting_but_rush_hour",
            when(not_commuting_rush, 1).when(not_commuting_rush.isNull(), 1).otherwise(0)
        )
        
        # C7: Police attendance or crime reference
        check_df = check_df.withColumn(
            "C7_police_attended_or_crime_reference",
            when((col("is_police_attendance") == True) | (col("is_crime_reference_provided") == True), 1).otherwise(0)
        )
        
        # C9: Policy inception within 30 days
        check_df = check_df.withColumn(
            "inception_to_claim", 
            datediff(to_date(col("start_date")), to_date(col("policy_start_date")))
        )
        
        check_df = check_df.withColumn(
            "C9_policy_within_30_days",
            when(col("inception_to_claim").between(0, 30), 1)
            .when(col("inception_to_claim").isNull(), 1)
            .otherwise(0)
        )
        
        # C10: Claim within 60 days of policy end
        check_df = check_df.withColumn(
            "claim_to_policy_end", 
            datediff(to_date(col("policy_renewal_date")), to_date(col("start_date")))
        )
        
        check_df = check_df.withColumn(
            "C10_claim_to_policy_end",
            when(col("claim_to_policy_end") < 60, 1)
            .when(col("claim_to_policy_end").isNull(), 1)
            .otherwise(0)
        )
        
        # C11: Young or inexperienced driver
        condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1)
        check_df = check_df.withColumn(
            "C11_young_or_inexperienced", 
            when(condition_inexperienced, 1)
            .when(condition_inexperienced.isNull(), 1)
            .otherwise(0)
        )
        
        # C12: Expensive vehicle for driver age
        condition_expensive_car = (
            ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | 
            ((col("age_at_policy_start_date_1") >= 25) & (col("vehicle_value") >= 30000))
        )
        
        check_df = check_df.withColumn(
            "C12_expensive_for_driver_age", 
            when(condition_expensive_car, 1)
            .when(condition_expensive_car.isNull(), 1)
            .otherwise(0)
        )
        
        # C14: Watchwords in circumstances
        watch_words = "|".join([
            "commut", "deliver", "parcel", "drink", "police", "custody", "arrest", 
            "alcohol", "drug", "station", "custody"
        ])
        
        check_df = check_df.withColumn(
            "C14_contains_watchwords",
            when(lower(col("circumstances")).rlike(watch_words), 1)
            .when(col("circumstances").isNull(), 1)
            .otherwise(0)
        )
        
        print("Check variables (C1-C14) created")
        return check_df
    
    def create_train_test_split(self, check_df):
        """
        Create train/test split with stratification on target variable.
        """
        # Apply date filter (from experimental notebooks)
        check_df = check_df.filter(col("reported_date") <= '2024-07-31')
        
        # Convert to pandas for sklearn train_test_split
        df_risk_pd = check_df.coalesce(1).toPandas()
        
        # Replace -1 with 0 in svi_risk for stratification
        df_risk_pd['svi_risk'] = df_risk_pd['svi_risk'].replace(-1, 0)
        
        # Split with stratification on svi_risk
        train_df, test_df = train_test_split(
            df_risk_pd, 
            test_size=0.3, 
            random_state=42, 
            stratify=df_risk_pd.svi_risk
        )
        
        # Tag datasets
        train_df['dataset'] = 'train'
        test_df['dataset'] = 'test'
        
        # Combine and convert back to Spark DataFrame
        combined_df_pd = pd.concat([test_df, train_df])
        check_df = self.spark.createDataFrame(combined_df_pd)
        
        print(f"Train set: {train_df.shape[0]} records, Test set: {test_df.shape[0]} records")
        return check_df
    
    def clean_data_types(self, check_df):
        """
        Clean and standardize data types.
        """
        # Convert boolean columns to integer
        boolean_columns = [
            "vehicle_unattended", "excesses_applied", "is_first_party", 
            "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
            "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance"
        ]
        
        for col_name in boolean_columns:
            if col_name in check_df.columns:
                check_df = check_df.withColumn(col_name, col(col_name).cast("integer"))
        
        # Convert decimal columns to float
        decimal_cols = ['outstanding_finance_amount', 'vehicle_value', 'voluntary_amount']
        for col_name in decimal_cols:
            if col_name in check_df.columns:
                check_df = check_df.withColumn(col_name, col(col_name).cast("float"))
        
        return check_df

    def fill_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Fill missing values in the DataFrame using a single, efficient aggregation pass.
        """
        # --- Define column groups for different fill strategies ---
        mean_fills = [
            "policyholder_ncd_years", "inception_to_claim", "veh_age", "business_mileage", 
            "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", 
            "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", 
            "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", 
            "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", 
            "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", 
            "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end"
        ]
        neg_fills = [
            "vehicle_unattended", "excesses_applied", "is_first_party", "first_party_confirmed_tp_notified_claim", 
            "is_air_ambulance_attendance", "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
            "veh_age_more_than_10", "damageScore", "areasDamagedMinimal", "areasDamagedMedium", "areasDamagedHeavy", 
            "areasDamagedSevere", "areasDamagedTotal", "police_considering_actions", "is_crime_reference_provided", 
            "ncd_protected_flag", "boot_opens", "doors_open", "multiple_parties_involved", "is_incident_weekend", 
            "is_reported_monday", "driver_age_low_1", "claim_driver_age_low", "licence_low_1"
        ]
        one_fills = [
            "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", "C5_is_night_incident", 
            "C6_no_commuting_but_rush_hour", "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
            "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords"
        ]
        string_fills = [
            'car_group', 'incidentDayOfWeekC', 'incidentMonthC', 'employment_type_abi_code_5', 
            'employment_type_abi_code_4', 'employment_type_abi_code_3', 'employment_type_abi_code_2', 
            'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 'overnight_location_abi_code', 
            'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 'impact_speed_unit', 
            'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 'front_severity', 
            'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 'left_severity', 
            'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
            'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 
            'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
            'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
            'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 
            'employment_type_abi_code_1', 'incident_day_of_week', 'reported_day_of_week'
        ]

        # --- Step 1: Calculate all statistics in a single aggregation ---
        # Filter lists to only include columns present in the DataFrame
        existing_mean_cols = [c for c in mean_fills if c in df.columns]

        # Create a list of aggregation expressions
        agg_expressions = [mean(c).alias(c) for c in existing_mean_cols]

        # Execute the single aggregation job if there are columns to process
        imputation_values = {}
        if agg_expressions:
            imputation_values = df.agg(*agg_expressions).collect()[0].asDict()

        # --- Step 2: Build the final fill map ---
        fill_map = {}
        
        # Add mean and mode values from the pre-calculated dictionary
        fill_map.update({c: imputation_values.get(c) for c in existing_mean_cols if imputation_values.get(c) is not None})
        
        # Add static fill values
        fill_map.update({c: -1 for c in neg_fills if c in df.columns})
        fill_map.update({c: 1 for c in one_fills if c in df.columns})
        fill_map.update({c: 'missing' for c in string_fills if c in df.columns})
        
        # --- Step 3: Apply all fills in a single transformation ---
        return df.fillna(fill_map)
    
    def save_processed_data(self, check_df):
        """
        Save processed data to MLStore catalog.
        """

        # Get dynamic table path
        table_path = self.get_table_path("single_vehicle_incident_checks", "claims_pol_svi", "mlstore")
        
        print(f"Saving processed data to: {table_path}")
        
        # Check if file path exists, if not create it
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.env_config['mlstore_catalog']}.single_vehicle_incident_checks")

        # Write with partitioning
        check_df.write.mode("overwrite").option("overwriteSchema", "true").partitionBy("dataset").saveAsTable(table_path)
        
        print("Data saved successfully")
    
    def run_preprocessing_pipeline(self, save_data=True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            save_data (bool): Whether to save the processed data to MLStore
            
        Returns:
            Spark DataFrame with processed data
        """
        print("Starting data preprocessing pipeline")
        
        # Load claim referral log
        clm_log = self.load_claim_referral_log()
        
        # Load SVI performance data and create target
        target_df = self.load_svi_performance_data(clm_log)
        
        # Get latest claim version
        latest_claim_version = self.get_latest_claim_version(target_df)
        
        # Load claim data
        check_df = self.load_claim_data(latest_claim_version)
        
        # Load policy data
        policy_svi = self.load_policy_data(check_df)
        
        # Deduplicate driver data
        check_df = self.deduplicate_driver_data(check_df)
        
        # Join claim and policy data
        check_df = self.join_claim_and_policy_data(check_df, policy_svi)
        
        # Calculate damage scores
        check_df = self.calculate_damage_scores(check_df)
        
        # Calculate vehicle and driver features
        check_df = self.calculate_vehicle_and_driver_features(check_df)
        
        # Create check variables (C1-C14)
        check_df = self.create_check_variables(check_df)
        
        # Clean data types
        check_df = self.clean_data_types(check_df)

        # Remove nan in column causing model training issue
        df = df.dropna(subset=['vehicle_overnight_location_id'])

        # Fill missing data
        check_df = self.fill_missing_values(check_df)
        
        # Create train/test split
        check_df = self.create_train_test_split(check_df)
        
        # Save if requested
        if save_data:
            self.save_processed_data(check_df)
        
        print("Data preprocessing pipeline completed successfully")
        return check_df