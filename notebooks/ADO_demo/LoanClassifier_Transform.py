# Databricks notebook source
# MAGIC %md
# MAGIC # Reading Data Lab
# MAGIC * The goal of this lab is to put into practice some of what you have learned about reading data with Azure Databricks and Apache Spark.
# MAGIC * The instructions are provided below along with empty cells for you to do your work.
# MAGIC * At the bottom of this notebook are additional cells that will help verify that your work is accurate.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 4)

# COMMAND ----------

# MAGIC %fs ls /mnt/training-sources/loan-data/loans

# COMMAND ----------

# DBTITLE 1,Read CSV
# CSV file read from mounted ADLS location
csv_file = "/mnt/training-sources/loan-data/loans/*.csv"
temp_df = (spark.read           # The DataFrameReader
   .option("header","true")
   .option("delimiter", "\t") 
   .csv(csv_file)               # Creates a DataFrame from CSV after reading in the file
)

loan_stats = temp_df

display(loan_stats)

# COMMAND ----------

print(loan_stats.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://kpistoropen.blob.core.windows.net/collateral/roadshow/logo_spark_tiny.png) Data Transformation
# MAGIC Run the following cell to transform columns in your `DataFrame` for easier analysis. Note: The transformations will not be applied until results are evaluated through `print` or `display` operation.

# COMMAND ----------

from pyspark.sql.functions import *

print("------------------------------------------------------------------------------------------------")
print("Create bad loan label, this will include charged off, defaulted, and late repayments on loans...")
loan_stats = loan_stats.filter(loan_stats.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))
loan_stats = loan_stats.withColumn("bad_loan", (~(loan_stats.loan_status == "Fully Paid")).cast("string"))


# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Turning string interest rate and revoling util columns into numeric columns...")
loan_stats = loan_stats.withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float')) 
loan_stats = loan_stats.withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float')) 
loan_stats = loan_stats.withColumn('issue_year',  substring(loan_stats.issue_d, 5, 4).cast('double') ) 
loan_stats = loan_stats.withColumn('earliest_year', substring(loan_stats.earliest_cr_line, 5, 4).cast('double'))
loan_stats = loan_stats.withColumn('credit_length_in_years', (loan_stats.issue_year - loan_stats.earliest_year))

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Converting emp_length column into numeric...")
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(loan_stats.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", "") ))
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(loan_stats.emp_length, "< 1", "0") ))
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(loan_stats.emp_length, "10\\+", "10") ).cast('float'))


# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Map multiple levels into one factor level for verification_status...")
loan_stats = loan_stats.withColumn('verification_status', trim(regexp_replace(loan_stats.verification_status, 'Source Verified', 'Verified')))



# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Calculate the total amount of money earned or lost per loan...")
loan_stats = loan_stats.withColumn('net', round( loan_stats.total_pymnt - loan_stats.loan_amnt, 2))

# COMMAND ----------

display(loan_stats)
