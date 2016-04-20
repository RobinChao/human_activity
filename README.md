## Human Activity Prediction Using Smartphones Data
Could you predict human behavior using smartphone accelerometer and gyroscope data?  We can try, using the [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) from UC Irvine.  

Accelerator and gyroscope data from a Samsung Galaxy S II smartphone was measured for 30 subjects performing six activites.  The raw smartphone data was processed into 561 variable columns.  Each subject repeated the activities over 50 times, resulting in over 10,000 rows of data.  Data was split into 21 subjects for training data and 9 subjects for test data.  The six activities were:
+ *WALKING*
+ *WALKING_UPSTAIRS*
+ *WALKING_DOWNSTAIRS*
+ *SITTING*
+ *STANDING*
+ *LAYING*

#### Prediction Methods
+ Prediction of human activity was done using a [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from scikit-learn.  
+ Prediction was also done using [Primary Component Analysis](http://scikit-learn.org/stable/modules/decomposition.html#pca) to reduce dimensionality before input to [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html#svm) Classifiers from scikit-learn.

#### Random Forest Optimization
Data exploration is given in __read_clean_data.py__.  Multiple column labels seemed redundant, and were reduced to 478 columns.  A grid search exploration of the maximum number of features at each split and number of estimators gave 85% to 90% prediction accuracy on the training set with validation.  Each parameter set was cross validated three times, showing some variation by max_features, though not always significant, and no variation for number of estimators.  From cross validation, I created boxplots to show the range of variation between data folds.  
<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_plots/gridscore_max_features.png" alt="Random Forest Score by Max Features at each split (max_features)" />
<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_plots/gridscore_n_estimators.png" alt="Random Forest Score by Number of Estimators (n_estimators)" />

The top ten important columns varied from one repetition to the next, with seven being consistently within them.  A near optimum was estimated at {*n_estimators 100*, *max_features 'sqrt'*}, and 90% prediction accuracy was confirmed on test data.  

A clear distinction between active (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS*) and sedentary (*SITTING, STANDING, LAYING*) activities can be made with histograms of several top importance variables.  

Script output is given in __read_clean_data.txt__ and plots in __human_activity_plots/__.

#### Random Forest Prediction
Further prediction was done with the full set of columns, given in __clean_predict_allvar.py__.  Train, validation and test data were reduced to a smaller set of subjects.  Classification parameters were explored more thoroughly in a grid search, with similar results: up to 90% prediction accuracy of validation data.  Variation is shown in a boxplot.  Near optimum parameters {'n_estimators': 50, 'max_features': 'log2'} were confirmed with test data, with 80% accuracy.  This is probably due to using fewer variable rows.

A cross-validation table and confusion matrix plot show a clear distinction between active (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS*) and sedentary (*SITTING, STANDING, LAYING*) activities.  However, there seems to be some correlation (up to 10%) within active activities, and more so (up to 50%) within sedentary activies.  Accuracy of each activity is between 60% and 90%.  The top ten columns account for 19.5% of total importance.  

Script output is given in __clean_predict_allvar.txt__ and plots in __human_activity_plots/__.

#### Prediction: PCA with SVM and Logistic Regression
Prediction using PCA as input to classifiers is given in __pca_clf.py__.   PCA dimensionality reduction was performed using all 562 columns.  Just the first 10 primary components account for 91% of explained variance ratio, while 100 components accounts for 98% of explained variance.  Using the first 30 components, representing 5.5% of the total, accounts for 95% of the explained variance, and seems a reasonable value for classifier input.  

Script output is give in __pca_clf_output.txt__ and plots in __human_activity_pca_plots/__.

Using PCA as input to Logistic Regression to fit training data gives reasonable accuracy (85%) with only 10 components, increasing with the number of PCA components.  30 PCA components gives 89% training data accuracy for all activities, with a standard error of 5% estimated by ten-fold cross-validation (due to variation between folds).  Test data prediction accuracy is within the training data fit margin of error.  

<img src="https://github.com/bfetler/human_activity/blob/master/human_activity_pca_plots/pca_lr.png" alt="Logistic Regression Score with Varying Number of PCA Components" />

Test data accuracy for each activity is between 85% and 93%, as shown by classification report.  Laying down may be more separable from the others due to inactivity, but a score of 100% is probably not reliable.  
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

Repeating the procedure using PCA as input to LinearSVC gives similar results.  

PCA with Logistic Regression or SVM seems to be faster and more accurate than Random Forest alone.  
