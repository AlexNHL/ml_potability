import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
df = pd.read_csv(r'\water_potability.csv')
data = pd.read_csv(r'\water_potability.csv')

# data cleaning 
df['ph'] = df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate'] = df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))
X = df.drop("Potability", axis=1)
Y = df["Potability"]


def split_train_test(data, test_ratio): # define a function to split train set and test set
    np.random.seed(10)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set_X, test_set_X = split_train_test(X, 0.2) # find the train set and test set of X
train_set_Y, test_set_Y = split_train_test(Y, 0.2) # find the train set and test set of Y

# Scaling
from sklearn.preprocessing import StandardScaler
Standardization_scaler = StandardScaler() 
standardization_train_set_X = Standardization_scaler.fit_transform(train_set_X) # use standardization to rescaled the data
standardization_test_set_X = Standardization_scaler.fit_transform(test_set_X) # use standardization to rescaled the data

# Support Vector Machine(SVM) classifier
from sklearn.svm import SVC
svm_clf = SVC(C = 1, gamma =0.1)
svm_clf.fit(standardization_train_set_X)

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
grid = GridSearchCV(svm_clf,param_grid,verbose = 1, cv=3, n_jobs = -1)
grid.fit(standardization_train_set_X,train_set_Y)
grid.best_params_ #Hyperparameter Tuning

svm_clf = SVC(C = 1, gamma =0.1)

from sklearn.model_selection import cross_val_predict 
Y_scores = cross_val_predict(svm_clf,standardization_train_set_X,train_set_Y, cv=10, method="decision_function") # roc 
from sklearn.metrics import roc_curve 
fpr, tpr, thresholds = roc_curve(train_set_Y, Y_scores) # find fpr and tpr and all thresholds

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid

plot_roc_curve(fpr, tpr)
plt.show()
#Error Analysis
from sklearn.metrics import roc_auc_score
roc_auc_score(train_set_Y, Y_scores)

from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from sklearn.model_selection import cross_val_predict # confusion_matrix
trian_pred_Y = cross_val_predict(svm_clf, standardization_train_set_X,train_set_Y, cv=10) 
recall_score(train_set_Y, trian_pred_Y)
precision_score(train_set_Y, trian_pred_Y)
accuracy_score(train_set_Y, trian_pred_Y)
f1_score(train_set_Y, trian_pred_Y)

#Error Analysis base on test set 
test_pred_Y = cross_val_predict(svm_clf, standardization_test_set_X, test_set_Y, cv=10) 
from sklearn.metrics import confusion_matrix
confusion_matrix(test_set_Y, test_pred_Y)


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=10)
from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
param_grid = {
    
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3 , 4],
    'min_samples_split': [6 ,8, 10, 12],
    'n_estimators': [200, 400, 600, 1000]
}
grid_search = GridSearchCV(estimator = forest_clf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1)

# Fit the grid search to the data
grid_search.fit(standardization_train_set_X,train_set_Y)
grid_search.best_params_ #Hyperparameter Tuning

forest_clf = RandomForestClassifier(random_state=10, max_depth = 80,
max_features =3,
min_samples_split = 8,
n_estimators = 600)
#Error Analysis
from sklearn.model_selection import cross_val_predict 
Y_scores = cross_val_predict(forest_clf , standardization_train_set_X,train_set_Y, cv=10,method="predict_proba") # roc 
Y_scores_forest = Y_scores[:, 1]
from sklearn.metrics import roc_curve 
fpr_forest, tpr_forest, thresholds = roc_curve(train_set_Y ,Y_scores_forest) # find fpr and tpr and all thresholds

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid

plot_roc_curve(fpr_forest, tpr_forest)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(train_set_Y,Y_scores_forest)


from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from sklearn.model_selection import cross_val_predict # confusion_matrix
trian_pred_Y = cross_val_predict(forest_clf, standardization_train_set_X,train_set_Y, cv=10) 
recall_score(train_set_Y, trian_pred_Y)
precision_score(train_set_Y, trian_pred_Y)
accuracy_score(train_set_Y, trian_pred_Y)
f1_score(train_set_Y, trian_pred_Y)

#Error Analysis base on test set 
test_pred_Y = cross_val_predict(forest_clf, standardization_test_set_X, test_set_Y, cv=10) 
from sklearn.metrics import confusion_matrix
confusion_matrix(test_set_Y, test_pred_Y)

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
Knn_clf = KNeighborsClassifier()
grid_params = { 'n_neighbors' : [1,4,9,16,25],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(Knn_clf, grid_params, verbose = 1, cv=3, n_jobs = -1)
gs.fit(standardization_train_set_X,train_set_Y)
gs.best_params_ #Hyperparameter Tuning

from sklearn.neighbors import KNeighborsClassifier
Knn_clf = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 25, weights = 'distance')
Knn_clf.fit(standardization_test_set_X,test_set_Y)

#Error Analysis
from sklearn.model_selection import cross_val_predict 
Y_scores = cross_val_predict(Knn_clf,standardization_train_set_X, train_set_Y, cv=10,method="predict_proba") # roc 
Y_scores_Knn = Y_scores[:, 1]
from sklearn.metrics import roc_curve 
fpr_Knn, tpr_Knn, thresholds_Knn = roc_curve(train_set_Y ,Y_scores_Knn) # find fpr and tpr and all thresholds

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid

plot_roc_curve(fpr_Knn, tpr_Knn)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(train_set_Y,Y_scores_Knn)

from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from sklearn.model_selection import cross_val_predict # confusion_matrix
trian_pred_Y = cross_val_predict(Knn_clf, standardization_train_set_X,train_set_Y, cv=10) 
recall_score(train_set_Y, trian_pred_Y)
precision_score(train_set_Y, trian_pred_Y)
accuracy_score(train_set_Y, trian_pred_Y)
f1_score(train_set_Y, trian_pred_Y)

test_pred_Y = cross_val_predict(Knn_clf, standardization_test_set_X, test_set_Y, cv=10) 
from sklearn.metrics import confusion_matrix
confusion_matrix(test_set_Y, test_pred_Y)


#GaussianProcessesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

model = GaussianProcessClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid = dict()
grid['kernel'] = [1*RBF(length_scale_bounds=(1e-5,1e5)), 1*DotProduct(length_scale_bounds=(1e-5,1e5)),
                  1*Matern(length_scale_bounds=(1e-5,1e5))]
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(standardization_train_set_X, train_set_Y)
results.best_estimator_ #Hyperparameter Tuning

Gpc = GaussianProcessClassifier(kernel=1**2 * RBF(length_scale=1))

from sklearn.model_selection import cross_val_predict 
Y_scores = cross_val_predict(Gpc,standardization_train_set_X,train_set_Y, cv=10, method="predict_proba") # roc 
Y_scores_Gpc = Y_scores[:, 1]
from sklearn.metrics import roc_curve 
fpr_Gpc, tpr_Gpc, thresholds_Gpc = roc_curve(train_set_Y, Y_scores_Gpc) # find fpr and tpr and all thresholds

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid

plot_roc_curve(fpr_Gpc, tpr_Gpc)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(train_set_Y, Y_scores_Gpc)

from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from sklearn.model_selection import cross_val_predict # confusion_matrix
trian_pred_Y = cross_val_predict(Gpc, standardization_train_set_X,train_set_Y, cv=10) 
from sklearn.metrics import confusion_matrix
confusion_matrix(train_set_Y, trian_pred_Y)
recall_score(train_set_Y, trian_pred_Y)
precision_score(train_set_Y, trian_pred_Y)
accuracy_score(train_set_Y, trian_pred_Y)
f1_score(train_set_Y, trian_pred_Y)

#Error Analysis base on test set 
test_pred_Y = cross_val_predict(Gpc, standardization_test_set_X, test_set_Y, cv=3) 
from sklearn.metrics import confusion_matrix
confusion_matrix(test_set_Y,test_pred_Y)

# roc cruve 
plot_roc_curve(fpr, tpr,"SVM")
plot_roc_curve(fpr_forest, tpr_forest,"Random Forest")
plot_roc_curve(fpr_Knn, tpr_Knn,"Knn")
plot_roc_curve(fpr_Gpc, tpr_Gpc, "Gpc")
plt.legend(loc = "lower right")
plt.show

# plot graphs of standardized parameters corresponding to Potable and NotPotable
import plotly.graph_objects as go
from plotly.colors import n_colors
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/water_potability.csv')

sc = StandardScaler()
df_1 = sc.fit_transform(df.iloc[:, :-1])
df_scaled = pd.DataFrame(df_1, columns = df.columns[:-1])
df_scaled['Potability'] = df['Potability']
columns = df.columns

#Potable
fig = go.Figure()
colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 9, colortype='rgb')
for idx, color in enumerate(colors):
    fig.add_trace(go.Violin(x=df_scaled[df_scaled['Potability']==1]
                              [columns[idx]]-idx/4, line_color=color, name=columns[idx], showlegend=False, hoverinfo='skip'))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(yaxis_showgrid=False, xaxis_zeroline=False, template='ggplot2', title_text='Distribution of features (Potability=1)')
fig.update_xaxes(showticklabels=False, title='Potable', showgrid=False)
fig.show()

#NotPotable
fig = go.Figure()
colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 9, colortype='rgb')
for idx, color in enumerate(colors):
    fig.add_trace(go.Violin(x=df_scaled[df_scaled['Potability']==0]
                              [columns[idx]]+idx/4, line_color=color, name=columns[idx], showlegend=False, hoverinfo='skip'))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(yaxis_showgrid=False, xaxis_zeroline=False, template='ggplot2', title_text='Distribution of features (Potability=0)')
fig.update_xaxes(showticklabels=False, title='Not Potable', showgrid=False)
fig.show()
