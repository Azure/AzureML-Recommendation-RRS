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
    # Query them in SQL
    import pydocumentdb.documents as documents
    import pydocumentdb.document_client as document_client
    import pydocumentdb.errors as errors
    import datetime

    MASTER_KEY = ''
    HOST = 'https://[].documents.azure.com:443/'
    DATABASE_ID = "recommendation_engine"
    COLLECTION_ID = "user_recommendations"
    database_link = 'dbs/' + DATABASE_ID
    collection_link = database_link + '/colls/' + COLLECTION_ID

    global client, collection
    client = document_client.DocumentClient(HOST, {'masterKey': MASTER_KEY})
    collection = client.ReadCollection(collection_link=collection_link)

def run(input_df):
    id = input_df.iloc[0][0]
    query = {'query': 'SELECT * FROM server s WHERE s.userId = ' + str(id)}

    options = {}

    result_iterable = client.QueryDocuments(collection['_self'], query, options)
    results = list(result_iterable);

    print(results)

    import json

    return json.dumps(str(results))
    
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
