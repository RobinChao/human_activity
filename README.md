## Human Activity Prediction Using Smartphones Data
Can we predict human physical activity using smartphone data, using smartphone accelerometer and gyroscope measurements?  In a study by researchers, the [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) is available from the UC Irvine Machine Learning Repository to help address this question.  

Accelerator and gyroscope data from a *Samsung Galaxy S II* smartphone was measured for 30 subjects performing six activites.  The smartphone data contained 561 variable columns, plus one column for the subject.  Most of the accelerometer and gyroscope data is reported as seemingly random variables in the range -1.0 to +1.0.  Each subject repeated the activities over 50 times, resulting in over 10,000 rows of data.  Data was split into 21 subjects for training data and 9 subjects for test data.  The six activities were:
+ *WALKING*
+ *WALKING_UPSTAIRS*
+ *WALKING_DOWNSTAIRS*
+ *SITTING*
+ *STANDING*
+ *LAYING*

The set of activity categories is fairly simple, but may serve as a template for more complex behavior.  Using the [Python Scikit-learn library](http://scikit-learn.org/stable/index.html), we were able to predict human behavior with reasonable accuracy.  

#### Prediction Methods
+ Prediction of human activity was done using a [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).  
+ Prediction was also done using [Primary Component Analysis](http://scikit-learn.org/stable/modules/decomposition.html#pca) to reduce dimensionality before input to [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html#svm) Classifiers, similar to the classic [eigenfaces](https://en.wikipedia.org/wiki/Eigenface) approach.

Both methods show that only 20 to 30 variables out of 500+ are needed to classify behavior.

#### Random Forest Optimization
Data exploration of training data is given in __read_clean_data.py__, with script output in __read_clean_data.txt__ and plots in __human_activity_plots/__.  Multiple column labels seemed duplicated or redundant, and were reduced to 478 columns.  A grid search exploration of the maximum number of features per split, and number of estimators (trees) gave 80% to 90% fit accuracy on the training set with validation.  Each parameter set was cross validated three times, showing some variation by *max_features*.  However, variation due to *max_features* was often not statistically significant (t-test p-value greater than 0.05), and fit score variation due to number of estimators (*n_estimators*) was also not significant.  Boxplots were created from cross validation scores, showing the range of variation within the data.  
<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_plots/gridscore_max_features.png" alt="Random Forest Score by Max Features at each split (max_features)" />
<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_plots/gridscore_n_estimators.png" alt="Random Forest Score by Number of Estimators (n_estimators)" />

Near optimum parameters were estimated at {*n_estimators: 100*, *max_features: 'sqrt'*}.   Prediction on test data with optimum parameters gave 90% accuracy.  

Test data accuracy for each activity is shown by classification report below.  Laying may be more separable from the others due to inactivity, but a score of 100% is probably not statistically reliable.  
<table>
<tr>
  <td><strong>Activity</strong></td>
  <td>Walking</td>
  <td>Walking Upstairs</td>
  <td>Walking Downstairs</td>
  <td>Sitting</td>
  <td>Standing</td>
  <td>Laying</td>
</tr>
<tr>
  <td><strong>Precision</strong></td>
  <td>0.83</td>
  <td>0.89</td>
  <td>0.95</td>
  <td>0.95</td>
  <td>0.92</td>
  <td>1.00</td>
</tr>
</table>

#### Random Forest Prediction
Further prediction was done with the full set of columns, given in __clean_predict_allvar.py__, with script output in __clean_predict_allvar.txt__ and plots in __human_activity_plots/__.  Train, validation and test data were reduced to a smaller set of subjects.  Classification parameters were explored more thoroughly in a grid search with cross-validation, with similar results: up to 90% prediction accuracy of validation data.  Variation is shown in a boxplot.  Near optimum parameters {*n_estimators: 50*, *max_features: 'log2'*} were used to predict test data with 80% accuracy.  

A cross-validation table and confusion matrix plot show a clear distinction between active (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS*) and sedentary (*SITTING, STANDING, LAYING*) activities, with some correlation within each group.  Prediction accuracy of each activity is between 60% and 90%.  

<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_plots/opt_conf_mat.png" alt="Confusion Matrix" />

The top ten columns account for 19% of total importance, and varied from one repetition to the next.  They typically include:
+ tGravityAcc_min_X
+ tGravityAcc_Mean_X
+ tGravityAcc_max_X
+ angle_X_gravityMean
+ tGravityAcc_energy_X
+ tGravityAcc_energy_Z
+ tGravityAcc_max_Z
+ angle_Z_gravityMean
+ tGravityAcc_Mean_Z
+ angle_Y_gravityMean

#### Prediction: PCA with SVM and Logistic Regression
Prediction using PCA as input to classifiers is given in __pca_clf.py__, with output in __pca_clf_output.txt__ and plots in __human_activity_pca_plots/__.   PCA dimensionality reduction was performed using all 562 variable columns.  Just the first 10 primary components account for 91% of explained variance ratio, while 100 components accounts for 98% of explained variance.  Using the first 30 components, representing 5.5% of the total 562 columns, accounts for 95% of the explained variance, and seems a reasonable value for classifier input.  

Using PCA as input to Logistic Regression to fit training data gives reasonable accuracy (85%) with only 10 components, increasing with the number of PCA components.  Using 30 PCA components gives 89% +- 5% training data accuracy for all activities, with a standard error estimated by 10x cross-validation due to variation in the data.  Test data prediction accuracy is within the training data fit margin of error.  

<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_pca_plots/pca_lr.png" alt="Logistic Regression Score with Varying Number of PCA Components" />

Test data accuracy for each activity is between 85% and 93%, as shown by classification report below.  Again, laying may be more separable from the others due to inactivity, but a score of 100% is probably not reliable.  
<table>
<tr>
  <td><strong>Activity</strong></td>
  <td>Walking</td>
  <td>Walking Upstairs</td>
  <td>Walking Downstairs</td>
  <td>Sitting</td>
  <td>Standing</td>
  <td>Laying</td>
</tr>
<tr>
  <td><strong>Precision</strong></td>
  <td>0.89</td>
  <td>0.93</td>
  <td>0.92</td>
  <td>0.91</td>
  <td>0.85</td>
  <td>1.00</td>
</tr>
</table>

Repeating the procedure using PCA as input to LinearSVC gives results similar to Logistic Regression.  

PCA with Logistic Regression or SVM seems to be faster and more accurate than Random Forest alone.  

#### Conclusion:
Can we predict human physical activity using smartphone data?  __Yes,__ within reasonable bounds for some simple activities.  
