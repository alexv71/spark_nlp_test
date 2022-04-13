import os
import sys
import csv
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, NGram, HashingTF, MinHashLSH, RegexTokenizer, SQLTransformer
import logging

from typing import Optional
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

if not os.path.isfile('./data/main_data.csv'):
    try:
        logging.info('Data file does not exist. Downloading data from UCI ML repository...')
        if not os.path.isdir('./data'):
            os.makedirs('./data')
        df = pd.read_excel (r'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
        df.dropna(subset=['Description'], inplace=True)
        df = df.groupby('Description').agg({
        'StockCode': 'first'
        }).reset_index()
        print(df)
        df.to_csv('./data/main_data.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)
    except Exception as e:
        logging.info('Exception happened: {}'.format(e))
        sys.exit(-1)

logging.info('Starting SPARK session...')
spark = SparkSession.builder.appName('pandas to spark').getOrCreate()

#stripYear = F.udf(lambda title: title[:-7])
transactions_df = (spark.read.csv('./data/main_data.csv', header=True, inferSchema=True)
    .drop('StockCode')
    #.withColumn('description', stripYear(F.col('description')))
)
#transactions_df.show(5, False)

logging.info('Starting SPARK modeling pipeline...')
model = Pipeline(stages=[
    SQLTransformer(statement="SELECT *, lower(Description) lower FROM __THIS__"),
    Tokenizer(inputCol="lower", outputCol="token"),
    StopWordsRemover(inputCol="token", outputCol="stop"),
    SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__"),
    RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1),
    NGram(n=2, inputCol="char", outputCol="ngram"),
    HashingTF(inputCol="ngram", outputCol="vector"),
    MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)
]).fit(transactions_df)
result_model = model.transform(transactions_df)
result_model = result_model.filter(F.size(F.col("ngram")) > 0)

def find_similarities(req_str):
    req_df = pd.DataFrame(columns=['description'])
    req_df.loc[0] = [req_str]
    result_request = model.transform(spark.createDataFrame(req_df))
    result_request = result_request.filter(F.size(F.col("ngram")) > 0)
    result = model.stages[-1].approxSimilarityJoin(result_request, result_model, 2.0, 'jaccardDist')
    result_df = result.select('datasetB.Description', 'jaccardDist').sort(F.col('jaccardDist')).toPandas().head(10)
    res = []
    for i in range(len(result_df)):
        res.append({
          'name': result_df.iloc[i]['Description'],
          'jaccard_dist': result_df.iloc[i]['jaccardDist']
        })
    logging.info('Request={}, returned TOP-{} results'.format(req_str, len(res)))
    return res


# Starting web application
logging.info('Starting web service...')
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Skupos"}

@app.get("/{req_str}")
def read_item(req_str: str):
    similarities = find_similarities(req_str)
    return {"result": similarities}
