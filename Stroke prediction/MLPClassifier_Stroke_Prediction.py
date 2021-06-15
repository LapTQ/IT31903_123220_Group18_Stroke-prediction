#FIRST IN-BUILT MACHINE LEARNING MODEL
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from numpy import nan
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
import time

dataset = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col="id")
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
target = 'stroke'
train_set, test_set = train_test_split(dataset, test_size=0.33, random_state=1)
train_set, test_set = train_set.copy(), test_set.copy()
n_uniques = dataset.nunique()
#determine percentage of unique value/number of values
for col in features:
    percentage = n_uniques[col] / dataset.shape[0] * 100
categorical_ix = []
numerical_ix = []
#determine numerical and categorical features
for col in features:
    percentage = n_uniques[col]/dataset.shape[0] * 100
    if percentage < 1:
        categorical_ix.append(col)
    else:
        numerical_ix.append(col)
#drop duplicate data
dups = train_set.duplicated()
n_dups = len(train_set[dups])
train_set.drop_duplicates(inplace=True)
#Remove outliers on numerical features - Quantile identifier
q25, q75 = train_set['bmi'].quantile(0.25), train_set['bmi'].quantile(0.75)
iqr = q75 - q25
factor = 4
cut_off = iqr * factor
bmi_upper = q75 + cut_off
age_lower = 1

mask_outlier = (train_set["bmi"] > bmi_upper) | (train_set["age"] < age_lower)
mask_nonoutlier = mask_outlier == False
train_set = train_set[mask_nonoutlier].copy()
##Remove outliers on numerical features - Standard Deviation identifier
#bmi_mean, bmi_std = train_set["bmi"].mean(), train_set["bmi"].std()
#factor = 4
#cut_off = bmi_std * factor
#bmi_upper = bmi_mean + cut_off
#age_lower = 1

#mask_outlier = (train_set["bmi"] > bmi_upper) | (train_set["age"] < age_lower)
#mask_nonoutlier = mask_outlier == False
#train_set = train_set[mask_nonoutlier].copy()
train_set["smoking_status"].replace('Unknown', nan, inplace=True)
test_set["smoking_status"].replace('Unknown', nan, inplace=True)
#Handle missing values by imputer
imp = SimpleImputer(strategy="most_frequent")
train_set_data = imp.fit_transform(train_set)
test_set_data = imp.fit_transform(test_set)
train_set = pd.DataFrame(train_set_data, index = train_set.index, columns = train_set.columns)
test_set = pd.DataFrame(test_set_data, index = test_set.index, columns = test_set.columns)
#prepare for data transformation
X_train, y_train = train_set[features].copy(), train_set[target].copy()
X_test, y_test = test_set[features].copy(), test_set[target].copy()
#NUMERICAL FEATURES ARE NORMALIZED TO THE RANGE (0, 1)
#CATEGORIAL FEATURES ARE ENCODED TO ORDINAL DATA
#There are other ways to transform
ct = ColumnTransformer([('scale', MinMaxScaler(), numerical_ix),
                        ('encode', OrdinalEncoder(), categorical_ix)],
                        remainder='passthrough')
ct.fit(X_train)
#setup encoder for target variablea
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
#perform transformation
X_train = ct.transform(X_train)
y_train = label_encoder.transform(y_train)
ct.fit(X_test)
label_encoder.fit(y_test)
X_test = ct.transform(X_test)
y_test = label_encoder.transform(y_test)
undersampling = RandomUnderSampler(random_state = 10)
X_train, y_train = undersampling.fit_resample(X_train, y_train)
    
model = MLPClassifier(hidden_layer_sizes = 200,
                          max_iter = 200)
start = time.time()
model.fit(X_train, y_train)
end =time.time()
print(end-start)
start = time.time()
predicted = model.predict(X_test)
end = time.time()
print(end-start)


#OKAYYYYYYYYYYYY NOW IT's MODEL TIMEEEEEEEE
#model = MLPClassifier(hidden_layer_sizes = 150,activation = 'tanh', max_iter = 150)
#model.fit(X_train, y_train)

#predicted = model.predict(X_test)
#report = classification_report(y_test, predicted)
#print(report)
def test_accuracy(hidden_layer_size, max_iters, X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_size,
                          max_iter = max_iters)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return f1_score(y_test, predicted)

def test_accuracy_SMOTE(hidden_layer_size, max_iters, X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_size,
                          max_iter = max_iters)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)    
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return f1_score(y_test, predicted)


def test_accuracy_Under(hidden_layer_size, max_iters, X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_size,
                          max_iter = max_iters)
    undersampling = RandomUnderSampler(random_state = 10)
    X_train, y_train = undersampling.fit_resample(X_train, y_train)    
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return f1_score(y_test, predicted)
f1_SMOTE = []
f1_Under = []
f1 = []
iarray = []
for i in range (1, 301, 10):
    plt.figure()
    plt.title("Epoch num:" + str(i), fontsize = 30) 
    plt.xlabel("Number of hidden layer")
    plt.ylim([0,100])
    plt.ylabel("Time")
    iarray.append(i)
    jarray = []
    time_none = []
    time_SMOTE = []
    time_Under = []
    f1.append([])
    f1_SMOTE.append([])
    f1_Under.append([])
    for j in range (1,301,10):
        start = time.time()
        f1[-1].append(test_accuracy(j, i, X_train, X_test, y_train, y_test))
        end = time.time() 
        time_none.append(end-start)
        start = time.time()  
        f1_SMOTE[-1].append(test_accuracy_SMOTE(j,i, X_train, X_test, y_train, y_test))
        end = time.time() 
        time_SMOTE.append(end-start)
        start = time.time()  
        f1_Under[-1].append(test_accuracy_Under(j,i, X_train, X_test, y_train, y_test))
        end = time.time() 
        time_Under.append(end-start)
        jarray.append(j)
    plt.plot(jarray,time_SMOTE,"^-r")
    plt.plot(jarray,time_Under,"v-b")
    plt.plot(jarray,time_none,"o-g")
    plt.show() 
plt.figure()    
heatmap_f1 = plt.imshow(f1, cmap = "autumn")
#for i in range(len(iarray)):
 #   for j in range(len(jarray)):
  #      plt.text(j, i, f1[i][j].round(), ha="center", va="center", color="w")
plt.colorbar(heatmap_f1)
plt.title("Heatmap origin")
plt.xlabel("Number Hidden Layer")
plt.ylabel("Number Max iteration")
plt.xticks(np.arange(len(iarray)), iarray)
plt.yticks(np.arange(len(jarray)), jarray)
plt.show()    

plt.figure()       
heatmap_f1_Under = plt.imshow(f1_Under[], cmap = "copper")
#for i in range(len(iarray)):
 #   for j in range(len(jarray)):
  #      plt.text(j, i, f1_Under[i][j].round(), ha="center", va="center", color="w")
plt.colorbar(heatmap_f1_Under)
plt.title("Heatmap UnderSampling")
plt.xlabel("Number Hidden Layer")
plt.ylabel("Number Max iteration")
plt.xticks(np.arange(len(iarray)), iarray)
plt.yticks(np.arange(len(jarray)),jarray)
plt.show()

plt.figure()       
heatmap_f1_SMOTE = plt.imshow(f1_SMOTE, cmap = "Greens")
#for i in range(len(iarray)):
 #   for j in range(len(jarray)):
  #      plt.text(j, i, f1_SMOTE[i][j].round(), ha="center", va="center", color="w")
plt.colorbar(heatmap_f1_SMOTE)
plt.title("Heatmap SMOTE")
plt.xlabel("Number Hidden Layer")
plt.ylabel("Number Max iteration")
plt.xticks(np.arange(len(iarray)), iarray)
plt.yticks(np.arange(len(jarray)),jarray)
plt.show()    
from sklearn.metrics import f1_score

undersampling = RandomUnderSampler(random_state = 10)
X_train, y_train = undersampling.fit_resample(X_train, y_train)    
activations = ["relu", "tanh", "logistic", "identity"]
solvers = ['lbfgs', 'sgd', 'adam']
plt.figure()
f1 = []
for i in solvers:
    model = MLPClassifier(hidden_layer_sizes = 200,
                          max_iter = 200, solver = i)
    model.fit(X_train, y_train) 
    predicted = model.predict(X_test)
    f1.append(f1_score(y_test, predicted))
x = np.arange(len(solvers))
fig, ax = plt.subplots()
ax.bar(x, f1, width = 0.35)
ax.set_title("Solver compare")
ax.set_ylabel("f1 score")
ax.set_xticks(x)
ax.set_xticklabels(solvers)
fig.tight_layout()
plt.show()
