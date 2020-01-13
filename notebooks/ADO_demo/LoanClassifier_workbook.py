# Databricks notebook source
# MAGIC %md 
# MAGIC # ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Bad Loan Prediction
# MAGIC 
# MAGIC Being able to accurately assess the risk of a loan application can save a lender the cost of holding too many risky assets. Rather than a credit score or credit history which tracks how reliable borrowers are, we will generate a score of how profitable a loan will be compared to other loans in the past. The combination of credit scores, credit history, and profitability score will help increase the bottom line for financial institution.  This lab will demonstrate how we can use Apache Spark ML to predict bad loans.  
# MAGIC 
# MAGIC Having a interporable model that an loan officer can use before performing a full underwriting can provide immediate estimate and response for the borrower and a informative view for the lender.
# MAGIC 
# MAGIC <a href="https://ibb.co/cuQYr6"><img src="https://preview.ibb.co/jNxPym/Image.png" alt="Image" border="0"></a>
# MAGIC 
# MAGIC 
# MAGIC This notebook is dependent on the original Loan Risk Analysis Python notebook and executes its binary classifier using a Gradient Boosting Tree (GBT) Algorithm.
# MAGIC 
# MAGIC Here are the SparkML <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html" target="_new">Python docs</a> and the <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.package" target="_new">Scala docs</a>.
# MAGIC  
# MAGIC Additional Resources:
# MAGIC * [MLlib Evaluation Metrics > Binary Classification](https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#binary-classification)
# MAGIC * [Binary Classification > MLlib Pipelines](https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html)
# MAGIC * [ML Tuning: model selection and hyperparameter tuning > Cross-Validation](https://spark.apache.org/docs/latest/ml-tuning.html#cross-validation)
# MAGIC 
# MAGIC For this lesson, we will use historical loans from Lending League.
# MAGIC 
# MAGIC In this lab:
# MAGIC * *Part 0*: Exploratory Analysis
# MAGIC * *Part 1*: ML Modeling
# MAGIC * *Part 2*: Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/wiki-book/general/logo_spark_tiny.png) *Part 0:* Exploratory Analysis
# MAGIC 
# MAGIC Let's start by taking a look at our data.  It's already mounted in `/mnt/training-sources/loan-data/loans/parquet/` table for us.  Exploratory analysis should answer questions such as:
# MAGIC 
# MAGIC * How many observations do I have?
# MAGIC * What are the features?
# MAGIC * Do I have missing values?
# MAGIC * What do summary statistics (e.g. mean and variance) tell me about my data?
# MAGIC 
# MAGIC Start by importing the data.  Bind it to `loan_data` by running the cell below

# COMMAND ----------

loan_data = spark.read.parquet("dbfs:/mnt/training-sources/training-data/")

loan_data.printSchema()

# COMMAND ----------

display(loan_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a count of the data using the `count()` DataFrame method.

# COMMAND ----------

loan_data.count()

# COMMAND ----------

training = loan_data.filter(loan_data.issue_year <= '2016')
valid = loan_data.filter(loan_data.issue_year > '2016')

# COMMAND ----------

training.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/wiki-book/general/logo_spark_tiny.png) *Part 1:* Evaluating Risk for Loan Approvals

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorical vs Numerics
# MAGIC 
# MAGIC - Categorical data represent characteristics such as a person’s gender, marital status, hometown, or the types of movies they like. Categorical data can take on numerical values (such as “1” indicating male and “2” indicating female), but those numbers don’t have mathematical meaning.  Therefore, our we must identify categorical data in order for our model to utilize it correctly.
# MAGIC 
# MAGIC - Numerical data can be broken into two types: discrete and continuous.
# MAGIC  - Discrete data represent items that can be counted; they take on possible values that can be listed out. The list of possible values may be fixed (also called finite); or it may go from 0, 1, 2, on to infinity (making it countably infinite). 
# MAGIC  
# MAGIC  - Continuous data represent measurements; their possible values cannot be counted and can only be described using intervals on the real number line. In this way, continuous data can be thought of as being uncountably infinite. For ease of recordkeeping, statisticians usually pick some point in the number to round off.

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

display(preppedDataDF)

# COMMAND ----------

splitYear = 2016
print("Taking all completed loans issued before {0} as training set.".format(splitYear))
train = preppedDataDF.filter(preppedDataDF.issue_year <= splitYear).cache()
print("Training set contains {} records".format(train.count()))

print("Taking all completed loans issued after {0} as training set.".format(splitYear))
valid = preppedDataDF.filter(preppedDataDF.issue_year > splitYear).cache()
print("Validation set contains {} records".format(valid.count()))

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

