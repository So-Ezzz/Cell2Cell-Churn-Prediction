#!/bin/bash

PROJECT_NAME="Cell2Cell-Churn-Prediction"

mkdir -p $PROJECT_NAME/{data/{raw,processed},notebooks,src,models,results,reports,scripts}

touch $PROJECT_NAME/README.md
touch $PROJECT_NAME/requirements.txt
touch $PROJECT_NAME/reports/final_report.md

touch $PROJECT_NAME/src/{__init__.py,config.py,data_loader.py,preprocess.py,train.py,evaluate.py,predict.py}

echo "Project structure created successfully."