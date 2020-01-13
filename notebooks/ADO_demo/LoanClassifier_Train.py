# Databricks notebook source
#Databricks Notebook

# COMMAND ----------

loan_data = spark.read.parquet("dbfs:/mnt/training-sources/training-data/")

loan_data.printSchema()

# COMMAND ----------

training = loan_data.filter(loan_data.issue_year <= '2016')
valid = loan_data.filter(loan_data.issue_year > '2016')

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier

print("------------------------------------------------------------------------------------------------")
print("Setting variables to predict bad loans")
myY = "bad_loan"
categoricals = ["term", "home_ownership", "purpose", "addr_state",
                "verification_status","application_type", "grade", "initial_list_status" ]
numerics = ["loan_amnt","emp_length", "annual_inc","dti", "installment",
            "revol_util","total_acc", "mort_acc","open_acc", "pub_rec","pub_rec_bankruptcies",
            "credit_length_in_years"]
myX = categoricals + numerics
print("Using following feature columns: {}".format(myX))
loan_stats2 = loan_data.select(myX + [myY, "int_rate","net", "issue_year"])

# Establish stages for our GBT model
indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals))
imputers = Imputer(inputCols = numerics, outputCols = numerics)
featureCols = list(map(lambda c: c+"_idx", categoricals)) + numerics

# Define vector assemblers
model_matrix_stages = indexers + \
                      [imputers] + \
                      [VectorAssembler(inputCols=featureCols, outputCol="features"), \
                       StringIndexer(inputCol="bad_loan", outputCol="label")]

partialPipeline = Pipeline().setStages(model_matrix_stages)
pipelineModel = partialPipeline.fit(loan_stats2)
preppedDataDF = pipelineModel.transform(loan_stats2)



# COMMAND ----------

# DBTITLE 1,Build GBT Classifier Model

# Define a GBT model.
gbt = GBTClassifier(featuresCol="features",
                    labelCol="label",
                    lossType = "logistic",
                    maxBins = 52,
                    maxIter=5,
                    maxDepth=5)
                    
# Train model.  This also runs the indexer.
gbt_model = gbt.fit(train)

predictions = gbt_model.transform(valid)



# COMMAND ----------

# DBTITLE 1,Evaluate Results using BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# DBTITLE 1,Perform Cross Validation by using Parameter Grid
# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 5])
             .addGrid(gbt.maxBins, [52])
             .addGrid(gbt.maxIter, [10, 20])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
cvModel = cv.fit(train)




# COMMAND ----------

# DBTITLE 1,Display Model Metrics on Validation Set
from pyspark.ml.evaluation import BinaryClassificationEvaluator

gbt_valid = cvModel.transform(valid)

# evaluate. note only 2 metrics are supported out of the box by Spark ML.
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
au_roc = bce.setMetricName('areaUnderROC').evaluate(gbt_valid)
au_prc = bce.setMetricName('areaUnderPR').evaluate(gbt_valid)

print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))


# COMMAND ----------

