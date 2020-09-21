FROM python:3.8

RUN pip install mlflow azure-storage-blob

ENV MLFLOW_SERVER_HOST 0.0.0.0
ENV MLFLOW_SERVER_PORT 5000

# Supply these at instanciation:
#ENV AZURE_STORAGE_ACCESS_KEY
#ENV MLFLOW_SERVER_FILE_STORE
#ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT

CMD mlflow server \
        --backend-store-uri $MLFLOW_SERVER_FILE_STORE \
        --default-artifact-root $MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT \
        --host $MLFLOW_SERVER_HOST \
        --port $MLFLOW_SERVER_PORT
