# This script generates the scoring and schema files
# Creates the schema, and holds the init and run functions needed to 
# operationalize the Iris Classification sample

# Import data collection library. Only supported for docker mode.
# Functionality will be ignored when package isn't found
try:
    from azureml.datacollector import ModelDataCollector
except ImportError:
    print("Data collection is currently only supported in docker mode. May be disabled for local mode.")


    # Mocking out model data collector functionality
    class ModelDataCollector(object):
        def nop(*args, **kw): pass

        def __getattr__(self, _): return self.nop

        def __init__(self, *args, **kw): return None


    pass

import os


# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.
def init():
    from azureml.dataprep import datasource
    df = datasource.load_datasource('ratings.dsource')

    from pyspark.ml.recommendation import ALS
    als = ALS() \
        .setUserCol("userId") \
        .setRatingCol("rating") \
        .setItemCol("movieId") \

    alsModel = als.fit(df)
    global userRecs
    userRecs = alsModel.recommendForAllUsers(10)

    # Query them in SQL
    import pydocumentdb.documents as documents
    import pydocumentdb.document_client as document_client
    import pydocumentdb.errors as errors
    import datetime

    MASTER_KEY = 'oX6tWPep8FCah8RM258s7cC3x9Kl8tWdbDxmNknXCP34ShW1Ag1ladvb5QWuBmMxuRISBO2HfrRFv3QeJYCSYg=='
    HOST = 'https://dcibrecommendationhack.documents.azure.com:443/'
    DATABASE_ID = "recommendation_engine"
    COLLECTION_ID = "user_recommendations"
    database_link = 'dbs/' + DATABASE_ID
    collection_link = database_link + '/colls/' + COLLECTION_ID

    global client, collection
    client = document_client.DocumentClient(HOST, {'masterKey': MASTER_KEY})
    collection = client.ReadCollection(collection_link=collection_link)

    # from pyspark.sql import SparkSession

    # spark = SparkSession.builder.getOrCreate()
    # userRecs = spark.read.parquet("./outputs/userrecs.parquet")

    # global inputs_dc, prediction_dc

    # inputs_dc = ModelDataCollector("./outputs/userrecs.parquet", identifier="inputs")
    # prediction_dc = ModelDataCollector("./outputs/userrecs.parquet", identifier="prediction")

    # from sklearn.externals import joblib

    # load the model file
    # global model
    # model = joblib.load('model.pkl')

    # inputs_dc = ModelDataCollector("model.pkl", identifier="inputs")
    # prediction_dc = ModelDataCollector("model.pkl", identifier="prediction")


def run(input_df):
    # from pyspark.sql.functions import col

    # pred = userRecs.filter(userRecs['userId'] > input_df.iloc[0][0])
    id = input_df.iloc[0][0]
    query = {'query': 'SELECT * FROM server s WHERE s.userId = ' + str(id)}

    options = {}

    result_iterable = client.QueryDocuments(collection['_self'], query, options)
    results = list(result_iterable);

    print(results)

    import json

    return json.dumps(str(results))

    # append 40 random features just like the training script does it.
    # import numpy as np
    # n = 40
    # random_state = np.random.RandomState(0)
    # n_samples, n_features = input_df.shape
    # input_df = np.c_[input_df, random_state.randn(n_samples, n)]
    # inputs_dc.collect(input_df)
    #
    # pred = model.predict(input_df)
    # prediction_dc.collect(pred)
    # return json.dumps(str(pred[0]))


def main():
    from azureml.api.schema.dataTypes import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services import generate_schema
    import pandas

    df = pandas.DataFrame(data=[[37]],
                          columns=['userId'])

    # Turn on data collection debug mode to view output in stdout
    os.environ["AML_MODEL_DC_DEBUG"] = 'true'

    # Test the output of the functions
    init()
    input1 = pandas.DataFrame([[3.0]])
    print("Result: " + run(input1))

    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

    # Genereate the schema
    generate_schema(run_func=run, inputs=inputs, filepath='./outputs/service_schema.json')
    print("Schema generated")


if __name__ == "__main__":
    main()
