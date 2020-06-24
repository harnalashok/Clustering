# Last amended: 28th May, 2020
# Data Source: Kaggle: https://www.kaggle.com/mohansacharya/graduate-admissions/kernels

# Objective:
#           i) Simple pandas data processing
#          ii) Clusteringf using kmeans
#         iii) Scree plots


# 1.0 Call libraries
%reset -f                       # Reset memory
# 1.1 Data manipulation library
import pandas as pd
import numpy as np
# 1.2 OS related package
import os
# 1.3 Modeling librray
# 1.3.1 Scale data
from sklearn.preprocessing import StandardScaler
# 1.3.2 Split dataset
from sklearn.model_selection import train_test_split
# 1.3.3 Class to develop kmeans model
from sklearn.cluster import KMeans
# 1.4 Plotting library
import seaborn as sns
# 1.5 How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# 1.6 Set numpy options to display wide array
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )


# 2.0 Set your working folder to where data is
os.chdir("D:\\data\\OneDrive\\Documents\\graduate_admissions")
os.chdir("/home/ashok/datasets/graduate_admissions")
os.listdir()

# 2.1 Read csv file
df = pd.read_csv("Admission_Predict_Ver1.2.csv.zip")

# 2.2 Explore dataset
# LOR: Letter of Recommendation Strength (out of 5)
# Research: Research Experience (0 or 1)
# CGPA: Undergraduate GPA (out of 10)
df.head()
df.shape               # (500, 10)
df.dtypes
df.columns.values      # 10 values
df.rename({'GRE.Score' : 'GRE_Score'},
           axis = 1,
           inplace = True
         )
# 2.3 How many get admitted/rejected
df.admit.value_counts()

# 2.3 Drop columns not needed.
#    Chance of Admit and admit columns contain the same information
df.drop(
        columns = ['Serial.No.', 'Chance of Admit '],    # List of columns to drop
        inplace = True                                   # Modify dataset here only
        )

df.columns              # 8 columns

# 3.0 Quick plots
sns.distplot(df.GRE_Score)                  # Distribution plot. Almost symmetric
sns.distplot(df.TOEFL_Score)                # Almost symmetric
sns.jointplot(df.GRE_Score, df.TOEFL_Score, kind = 'reg')   # Strong correlation
sns.jointplot(df.CGPA, df.GRE_Score,        kind = 'reg')   # Strong correlation (0.83)
sns.catplot('admit','GRE_Score', data = df, kind = 'box')      # Relationship of 'admit' to GRE_SCore
pd.plotting.andrews_curves(df,
                           'admit',
                           colormap = 'winter'       # Is there any pattern in the data?
                           )

sns.barplot('admit', 'GRE_Score',   estimator = np.mean, data = df) # Avg GRE score of admit vs non-admit
sns.barplot('admit', 'TOEFL_Score', estimator = np.mean, data = df)

# 3.1 Copy 'admit' column to another variable and then drop it
#     We will not use it in clustering
y = df['admit'].values
df.drop(columns = ['admit'], inplace = True)


# 3.2 Scale data using StandardScaler
ss = StandardScaler()     # Create an instance of class
ss.fit(df)                # Train object on the data
X = ss.transform(df)      # Transform data
X[:5, :]                  # See first 5 rows


# 4.0 Split dataset into train/test
X_train, X_test, _, y_test = train_test_split( X,               # np array without target
                                               y,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# 4.1 Examine the results
X_train.shape              # (375, 7)
X_test.shape               # (125, 7)

# 5.0 Develop model
# 5.1 Create an instance of modeling class
#     We will have two clusters
"""
Uses kmeans++ initialization
Ref: https://www.geeksforgeeks.org/ml-k-means-algorithm/
kmeans++ steps for initialization involved are:

        1) Randomly select the first centroid from the data points.
        2) For each data point compute its distance from the nearest,
           previously choosen centroid.
        3) Select the next centroid from the data points such that the
           probability of choosing a point as centroid is directly 
           proportional to its distance from the nearest, previously
           chosen centroid. (i.e. the point having maximum distance
           from the nearest centroid is most likely to be selected 
           next as a centroid)
        4) Repeat steps 2 and 3 untill k centroids have been sampled

Kmeans algorith used is 'elkan' which is a modification of Lloyd's
algorithm. (https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
For Vornoi diagram, please see:
http://cs.brown.edu/courses/cs252/misc/resources/lectures/pdf/notes09.pdf    

"""

clf = KMeans(n_clusters = 2)
# 5.2 Train the object over data
clf.fit(X_train)

# 5.3 So what are our clusters?
clf.cluster_centers_
clf.cluster_centers_.shape         # (2, 7)
clf.labels_                        # Cluster labels for every observation
clf.labels_.size                   # 375
clf.inertia_                       # Sum of squared distance to respective centriods, SSE
# 5.4 For importance and interpretaion of silhoutte score, see:
# See Stackoverflow:  https://stats.stackexchange.com/q/10540
silhouette_score(X_train, clf.labels_)    # 0.20532663345078295


# 6 Make prediction over our test data and check accuracy
y_pred = clf.predict(X_test)
y_pred
# 6.1 How good is prediction
np.sum(y_pred == y_test)/y_test.size

# 7.0 Are clusters distiguisable?
#     We plot 1st and 2nd columns of X
#     Each point is coloured as per the
#     cluster to which it is assigned (y_pred)
dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)


# 7.1 Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # 7.1.1 How many clusters?
    n_clusters = i+1
    # 7.1.2 Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    # 7.1.3 Train the kmeans object over data
    clf1.fit(X_train)
    # 7.1.4 Store the value of inertia in sse
    sse.append(clf1.inertia_ )

# 7.2 Plot the line now
sns.lineplot(range(1, 11), sse)

# 8.0 Silhoutte plot
# Ref: https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
"""
Higher the Silhoutte coef of each point, better is clustering.
In the diagram, for each cluster, for various points, Silhoutte
coef reduces rapidly. Each cluster graph should be as flat as
possible at the edge. See the silhoutee of four points below:

Good Silhoutte
----------------------          pt1   This coef should be very high
----------------------          pt2
----------------------          pt3
-------------------             pt4

Bad Silhoutte
----------------------          pt1
---------------                 pt2
--------                        pt3
---                             pt4   Silhoutte coef very less

"""


visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure

# Intercluster distance: Does not work
from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure



########################
# Permutation importance vs Feature Importance
# Prefer Permutation importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification

# Generate some data
X,y = make_classification(n_samples = 1000)

X.shape
y.shape
y[:10]

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.33)
X_train.shape
X_test.shape

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

result = permutation_importance( clf,X_train,y_train)
indx = (-result.importances_mean).argsort()         # Ascending order
indx

indx_rf = (-clf.feature_importances_).argsort()
indx_rf
# How many matches
indx == indx_rf
#####################################################

# Para 6.19.1, Page 1310
# Impute missing values with variants of IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
