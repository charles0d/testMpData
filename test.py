import flask
from flask import request, jsonify
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
import copy

def score_classifier(dataset,classifier,labels):
    
    """
        performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
        :param dataset: the dataset to work on
        :param classifier: the classifier to use
        :param labels: the labels used for training and validation
        :return: the fscore and prediction function of the fitted classifier averaged on the 3 folds
        """
    
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    fscore = 0
    clf_list = []
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        clf_list.append(copy.deepcopy(classifier))
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        fscore += f1_score(test_labels, predicted_labels)
    recall/=3
    fscore/=3
    #print(confusion_mat)
    #print(fscore)
    
    # return the score and also the fitted classifier averaged on the folds
    return fscore, lambda x: (np.mean([clf.predict(x) for clf in clf_list], axis=0) > .5).astype(int)

# Load dataset
df = pd.read_csv("nba_logreg.csv")

# extract names, labels, features names and values
names = df['Name'].values.tolist() # players names
labels = df['TARGET_5Yrs'].values # labels
paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1).values

# replacing Nan values (only present when no 3 points attempts have been performed by a player)
for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x]=0.0

# normalize dataset : set all values between 0 and 1
mmscaler = MinMaxScaler()
X = mmscaler.fit_transform(df_vals)

#set a random_state
rs=50

# split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.3,random_state=rs,shuffle=True,stratify=labels)
clf_list = [] # will store the chosen (unfitted but with optimized hyperparams) classifiers
pred_list = [] # will store the chosen (trained on X_train) prediction functions returned by score_classifier



####
#### Train the models on the train dataset, using score_classifier
####

## Logistic Regression
## We test for l2 penalization, and C between 0.01 and 100
Cs = np.logspace(-2,2,num = 10)
C_best,score_best = 0,0
logReg_pred = None # Will store the optimized model
logReg_best = None # Will store the fitted optimized classifier
for c in Cs:
    logReg = LogisticRegression(solver = 'lbfgs',C=c,random_state=rs,max_iter=1000)
    f,pred = score_classifier(X_train,logReg,y_train)
    if f>score_best:
        C_best,score_best = c,f
        logReg_best = logReg
        logReg_pred = pred

clf_list.append(logReg_best)
pred_list.append(logReg_pred)

##
## Now the KNN model : we test between 3 and 30 neighbors
neighs = np.arange(3,30,2)
k_best,score_best = 0,0
kNN_pred = None
kNN_best = None
for k in neighs:
    kNN = KNeighborsClassifier(n_neighbors = k)
    f,pred = score_classifier(X_train,kNN,y_train)
    if f>score_best:
        k_best,score_best = k,f
        kNN_pred = pred
        kNN_best = kNN

clf_list.append(kNN_best)
pred_list.append(kNN_pred)

##
## Now the SVC, we'll use the rbf kernel and tune gamma and C looking between 0.01 and 100 for both
gammas = np.logspace(-2,2,10)
Cs = np.logspace(-2,2,10)
gamma_best,C_best,score_best = 0,0,0
SVC_pred = None
SVC_best = None
for gamma in gammas:
    for C in Cs:
        svc = SVC(C=C,gamma=gamma,random_state=rs)
        f,pred = score_classifier(X_train,svc,y_train)
        # remember only the best model (/ parameters)
        if(f > score_best):
            gamma_best,C_best,score_best = gamma,C,f
            SVC_pred = pred
            SVC_best = svc

clf_list.append(SVC_best)
pred_list.append(SVC_pred)

## Now the random forest
## We'll optimize over the number of estimators and max_depth, leaving other parameters as default
n_ests = np.arange(50,501,100)
max_depths = [2,5,10,None]
nest_best,mdepth_best,score_best = 0,0,0
rf_pred = None
rf_best = None
for n_est in n_ests:
    for i,max_depth in enumerate(max_depths):
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=rs)
        f,pred = score_classifier(X_train,rf,y_train)
        if f>score_best:
            nest_best,mdepth_best,score_best = n_est,max_depth,f
            rf_pred = pred
            rf_best = rf

clf_list.append(rf_best)
pred_list.append(rf_pred)

##
## Last, the Gradient Boosting. We'll optimize over n_estimators and max_features
n_ests = np.arange(50,501,100)
max_feats = [2,4,6,10,None]
nest_best,mfeat_best,score_best = 0,0,0
gb_pred = None
gb_best = None
for n_est in n_ests:
    for i,max_feat in enumerate(max_feats):
        gb = GradientBoostingClassifier(n_estimators=n_est,max_features=max_feat,random_state=rs)
        f,pred = score_classifier(X_train,gb,y_train)
        if(f > score_best):
            nest_best,mfeat_best,score_best = n_est,max_feat,f
            gb_pred = pred
            gb_best = gb

clf_list.append(gb_best)
pred_list.append(gb_pred)


## Finally, train the optimized classifiers on the whole dataset
final_pred_list = []
for clf in clf_list:
    f,pred = score_classifier(X,clf,labels)
    final_pred_list.append(pred)

# final prediction : majority voting.
# For 5 elements it comes down to integer-dividing by 3
final_pred = lambda x: sum([pred(x) for pred in final_pred_list])//3


###
### Now the webservice part
###

# remove special characters and spaces from feature names
stdparamset = [param.replace('%', '').replace(' ', '') for param in paramset]

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/predict', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    inputs = []
    
    # We ask for all the features to be given
    for param in stdparamset:
        if param in request.args:
            inputs.append(float(request.args[param]))
        else:
            return f"Error: No {param} field provided. Please specify a {param}."

    sample = np.expand_dims(inputs, 0)
    sample = mmscaler.transform(sample)

    res = int(final_pred(sample)[0])
    
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify({"class": res})

app.run()


