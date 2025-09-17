import databricks.lakehouse_monitoring as lm
import datetime
from pyspark.sql.functions import lit
import time
from pyspark.sql import SparkSession

# Create a Spark session (assuming this is setup elsewhere in your code as `spark`)
spark = SparkSession.builder.appName("Delta Table Management").getOrCreate()


def check_and_create_table(df, table_name):
    """
    Checks if a Delta table exists in the Spark session. If it does not exist, creates the table.

    Parameters:
    - df: The DataFrame to write to the table if the table does not exist.
    - table_name: The name of the table to check and potentially create.
    """
    try:
        # Attempt to read the table to check if it exists
        spark.read.table(table_name)
        print(f"Table {table_name} already exists, skipping creation.")
    except Exception as e:
        # If an exception is caught, it indicates the table does not exist, so create it
        print(f"Table {table_name} does not exist, creating now.")
        df.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .saveAsTable(table_name)

# Function to check the refresh status
def check_refresh_status(refresh_id, TABLE_NAME):
    while True:
        run_info = lm.get_refresh(table_name=TABLE_NAME, refresh_id=refresh_id)
        if run_info.state in (lm.RefreshState.PENDING, lm.RefreshState.RUNNING):
            print("Waiting...")
            time.sleep(30)
        else:
            return run_info
        
# Function to check if the monitoring tag is set
def check_monitoring_tag(table_name):
    try:
        # Retrieve table properties
        properties = spark.sql(f"DESCRIBE TABLE EXTENDED {table_name}").collect()
        # Check for 'monitoring' property in the result
        for row in properties:
            if 'monitoring' in row.asDict().get('col_name', '') and 'enabled' in row.asDict().get('data_type', ''):
                return True
        return False
    except Exception as e:
        print(f"Error checking properties for {table_name}: {str(e)}")
        return False
    
def check_table_property(spark, table_name, property_name):
    """
    Checks a property on a specified table and returns its value.

    Args:
    spark (SparkSession): The active Spark session.
    table_name (str): The full name of the table.
    property_name (str): The name of the property to check.

    Returns:
    str: The value of the property or None if the property does not exist.
    """
    describe_command = f"DESCRIBE TABLE EXTENDED {table_name}"
    properties_df = spark.sql(describe_command)
    property_value = None
    for row in properties_df.collect():
        if row['col_name'] == 'Table Properties':
            # Split the properties string into a dictionary
            properties_dict = dict(item.split("=") for item in row['data_type'].strip('[]').split(","))
            property_value = properties_dict.get(property_name)
            break
    return property_value

import time

def create_and_monitor_inference_table(spark, lm, unityCatalog_catalog, unityCatalog_schema, table_name, 
                             monitoring_modelInferenceMonitoring_problemType, monitoring_modelInferenceMonitoring_predictionColumn, monitoring_modelInferenceMonitoring_timestampColumn, monitoring_modelInferenceMonitoring_granularities, monitoring_modelInferenceMonitoring_modelIdColumn, monitoring_modelInferenceMonitoring_groundTruth, today_date):
    """
    Create and monitor a table in Databricks.

    Args:
        spark: The active Spark session.
        lm: The monitor management object (or library) for creating and managing monitors.
        table_name: The name of the table to monitor.
        monitoring_status_checker: A callable that checks if monitoring is enabled.
        check_refresh_status: A callable that checks the status of a refresh.
        inference_log_params: Dictionary containing parameters for creating the inference log.
        output_schema_name: The schema to which the monitor outputs.
        today_date: The current date (string).
    """
    # Check if monitoring is enabled for the given table
    monitoring_status = check_table_property(spark, table_name, 'monitoring')
    proceed_with_creation = monitoring_status != 'enabled'

    if proceed_with_creation:
        # Create the monitor if it is not enabled
        info = lm.create_monitor(
            table_name=table_name,
            profile_type=lm.InferenceLog(
                problem_type=monitoring_modelInferenceMonitoring_problemType,
                prediction_col=monitoring_modelInferenceMonitoring_predictionColumn,
                timestamp_col=monitoring_modelInferenceMonitoring_timestampColumn,
                granularities=monitoring_modelInferenceMonitoring_granularities,
                model_id_col=monitoring_modelInferenceMonitoring_modelIdColumn,
                label_col=monitoring_modelInferenceMonitoring_groundTruth
            ),
            output_schema_name=f"{unityCatalog_catalog}.{unityCatalog_schema}"
        )
        print(f"Created monitor for {table_name}")

        # Wait for monitor to be created and become active
        while info.status == lm.MonitorStatus.PENDING:
            info = lm.get_monitor(table_name=table_name)
            time.sleep(10)
            print("Waiting for monitor to become active...")
        assert info.status == lm.MonitorStatus.ACTIVE, "Monitor did not become active."

        # Trigger and check refresh
        refreshes = lm.list_refreshes(table_name=table_name)
        assert len(refreshes) > 0, "No refreshes found."
        run_info = refreshes[0]  # Get the most recent refresh info
        run_info = check_refresh_status(run_info.refresh_id, table_name)

        # If the refresh failed, retry the refresh
        if run_info.state == lm.RefreshState.FAILED:
            print("Initial refresh failed, attempting to refresh again...")
            new_refresh = lm.run_refresh(table_name=table_name)
            run_info = check_refresh_status(new_refresh.refresh_id, table_name)

            if run_info.state == lm.RefreshState.FAILED:
                raise Exception("Refresh failed after retry.")
                spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('monitoring' = 'disabled')")
            elif run_info.state == lm.RefreshState.SUCCESS:
                print("Refresh succeeded after retry.")
        else:
            assert run_info.state == lm.RefreshState.SUCCESS, "Refresh did not succeed."

        # Update table properties indicating that monitoring is enabled
        spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('monitoring' = 'enabled', 'created_date' = '{today_date}')")
        print(f"Properties set for table {table_name}")
    else:
        print(f"Monitoring dashboard for {table_name} is already created.")

def create_and_monitor_data_table(spark, lm, unityCatalog_catalog, unityCatalog_schema, table_name, today_date):
    """
    Create and monitor a table in Databricks.

    Args:
        spark: The active Spark session.
        lm: The monitor management object (or library) for creating and managing monitors.
        table_name: The name of the table to monitor.
        monitoring_status_checker: A callable that checks if monitoring is enabled.
        check_refresh_status: A callable that checks the status of a refresh.
        inference_log_params: Dictionary containing parameters for creating the inference log.
        output_schema_name: The schema to which the monitor outputs.
        today_date: The current date (string).
    """
    # Check if monitoring is enabled for the given table
    monitoring_status = check_table_property(spark, table_name, 'monitoring')
    proceed_with_creation = monitoring_status != 'enabled'

    if proceed_with_creation:
        # Create the monitor if it is not enabled
        info = lm.create_monitor(
        table_name= table_name,
        profile_type=lm.Snapshot(),
        # schedule=lm.MonitorCronSchedule(
            #     quartz_cron_expression="0 0 12 * * ?", # schedules a refresh every day at 12 noon
            #     timezone_id="PST",
            # ),
        output_schema_name=f"{unityCatalog_catalog}.{unityCatalog_schema}"
        )
        print(f"Created monitor for {table_name}")

        # Wait for monitor to be created and become active
        while info.status == lm.MonitorStatus.PENDING:
            info = lm.get_monitor(table_name=table_name)
            time.sleep(10)
            print("Waiting for monitor to become active...")
        assert info.status == lm.MonitorStatus.ACTIVE, "Monitor did not become active."

        # Trigger and check refresh
        refreshes = lm.list_refreshes(table_name=table_name)
        assert len(refreshes) > 0, "No refreshes found."
        run_info = refreshes[0]  # Get the most recent refresh info
        run_info = check_refresh_status(run_info.refresh_id, table_name)

        # If the refresh failed, retry the refresh
        if run_info.state == lm.RefreshState.FAILED:
            print("Initial refresh failed, attempting to refresh again...")
            new_refresh = lm.run_refresh(table_name=table_name)
            run_info = check_refresh_status(new_refresh.refresh_id, table_name)

            if run_info.state == lm.RefreshState.FAILED:
                raise Exception("Refresh failed after retry.")
                spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('monitoring' = 'disabled')")
            elif run_info.state == lm.RefreshState.SUCCESS:
                print("Refresh succeeded after retry.")
        else:
            assert run_info.state == lm.RefreshState.SUCCESS, "Refresh did not succeed."

        # Update table properties indicating that monitoring is enabled
        spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('monitoring' = 'enabled', 'created_date' = '{today_date}')")
        print(f"Properties set for table {table_name}")
    else:
        print(f"Monitoring dashboard for {table_name} is already created.")