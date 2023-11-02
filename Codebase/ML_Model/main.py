# Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("Training.csv")
df_test = pd.read_csv("Testing.csv")

encoder = LabelEncoder()
df["prognosis"] = encoder.fit_transform(df["prognosis"])

def processed_df(df):
    df.dropna(axis = 1,inplace = True)
    encoder = LabelEncoder()
    df["prognosis"] = encoder.fit_transform(df["prognosis"])
    return df

def split_df(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:, -1]
    return X,y

def trained_model(X,y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def save_model():
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

def model_accuracy(model,X_test,y_test):
    preds = model.predict(X_test)
    print(f"Accuracy on test data by Random Forest Classifier\
    : {accuracy_score(y_test, preds)*100}")

def build_symptom_dict(df):
    X = X = df.iloc[:,:-1]
    symptoms = X.columns.values

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }
    return data_dict


# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models
def predictDisease(symptoms,data_dict1,model):
#     symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict1["symptom_index"])
    for symptom in symptoms:
        index = data_dict1["symptom_index"][symptom]
        input_data[index] = 1

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    prediction = data_dict1["predictions_classes"][model.predict(input_data)[0]]
    
    # making final prediction by taking mode of all predictions
    predictions = {
        "model": prediction,
    }
    return predictions

if __name__ == '__main__':
    df = processed_df(df)
    X,y = split_df(df)
    model = load_model("model.sav")
    df_test = processed_df(df_test)
    X_test,y_test = split_df(df_test)
    model_accuracy(model,X_test,y_test)
    data_dict = build_symptom_dict(df)
    print(predictDisease(["Irritability","Depression"], data_dict, model))