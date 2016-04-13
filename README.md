# human_activity

#### Human Activity by Smartphone Data Set
[Human Activity Recognition Using Smartphones Data](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) is available from UC Irvine for Machine Learning tasks.  Accelerator and gyroscope data from a Samsung Galaxy S II smartphone was measured for 30 subjects performing six activites (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING*).  The raw smartphone data was processed into 561 variable columns.  Each subject repeated the activities over 50 times, resulting in over 10,000 rows of data.  Data was split into 21 subjects for training data and 9 subjects for test data.  

#### Prediction by Random Forest Classification
Prediction of human activity was done using a [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from scikit-learn.  

#### Exploration
Data exploration is given in __read_clean_data.py__.  Multiple column labels seemed redundant, and were reduced to 478 columns.  Classification parameters were explored using a grid search, giving 85% to 90% prediction accuracy on a validation set.  Each parameter set was repeated several times, showing little variation between many of them, as shown in boxplots.  The top ten important columns varied from one repetition to the next, with seven being consistently within them.  A near optimum was estimated at {*n_estimators 100*, *max_features 'sqrt'*}, and 90% prediction accuracy was confirmed on test data.  

A clear distinction between active (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS*) and sedentary (*SITTING, STANDING, LAYING*) activities can be made with histograms of several top importance variables.  

Script output is given in __read_clean_data.txt__ and plots in __human_activity_plots/__.

#### Prediction
Further prediction was done with the full set of columns, given in __clean_predict_allvar.py__.  Train, validation and test data were reduced to a smaller set of subjects.  Classification parameters were explored more thoroughly in a grid search, with similar results: up to 90% prediction accuracy of validation data.  Variation is shown in a boxplot.  Near optimum parameters {'n_estimators': 50, 'max_features': 'log2'} were confirmed with test data, with 80% accuracy.  This is probably due to using fewer variable rows.

A cross-validation table and confusion matrix plot show a clear distinction between active (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS*) and sedentary (*SITTING, STANDING, LAYING*) activities.  However, there seems to be some correlation (up to 10%) within active activities, and more so (up to 50%) within sedentary activies.  

Script output is given in __clean_predict_allvar.txt__ and plots in __human_activity_plots/__.
