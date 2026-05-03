import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Loading the wine dataset
wine=load_wine()
X=wine.data
y=wine.target
# To view the dataset
df=pd.DataFrame(X,columns=wine.feature_names)
df['target']=y
# print(df.head())
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.10,random_state=42)
#Params for the RandomForest Model
max_depth=10
n_estimators=20
#Autologging 
mlflow.autolog()
#Mentioning your experiment below
mlflow.set_experiment('MLOPs-Experiment1')
# MLFlow Code
# with mlflow.start_run(experiment_id='id of the created experiment at the mlflow ui'):
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,random_state=42)
    rf.fit(X_train, y_train)
    y_pred=rf.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion_Matrix.png")
    #Logging
    ''' We do not need to log the metrics, params, artifacts using mlflow.autolog()
    But it does not logs the file/code and the tags
    '''
    # mlflow.log_metric("Accuracy",acc)
    # mlflow.log_param("Max_Depth", max_depth)
    # mlflow.log_param("N-estimators", n_estimators)
    # mlflow.log_artifact("Confusion_Matrix.png")
    mlflow.log_artifact(__file__)
    #Logging Tags
    mlflow.set_tags({
        "Author":"Imran",
        "Project":"Wine Classification"
    })
    #Logging Model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    print(acc)