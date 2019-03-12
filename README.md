# Yelp-Business-Rating-Prediction

## Problem Statement
The goal of this project is to predict business’s star rating using the reviews of that business and
review count based on neural network implementation in Tensorflow.

## Methodology
To implement this problem, we used two datasets provided by Yelp Dataset Challenge, first is
yelp_academic_dataset_review and second is yelp_academic_dataset_business. As these files are
in json format, we converted it into csv. The review file contains reviews for different businesses. We
grouped reviews by business_id to merge it with business file.
The dataset required preprocessing for reviews and review_counts. For reviews, we applied TF-IDF
to vectorize data and removed stop words. For review_counts, we did feature normalization to
convert its data in range 0 to 1.
The input for the models, that is ‘x’ consists of review_counts and TF-IDF vectors for reviews and
the output for the models, that is ‘y’ consists of stars. For Regression model, stars are used without
any preprocessing on it. While for Classification model, label encoding and then one-hot encoding is
applied on stars.
To implement Regression model using Tensorflow regression neural network model and to
implement Classification model using Tensorflow classification neural network model, we did
various experiments on number of neurons, number of hidden layers, using different activation
functions like relu, sigmoid and tanh, using different optimizers like adam, sgd and rmsprop. After
various experiments on number of neurons and hidden layers, we compared the selected to work
with two hidden layers with 50 neurons in first layer and 25 in second hidden layer. Using this
setting, we trained models in all combinations of activation functions and optimizers for both
Regression and Classification models. For each model, we used EarlyStopping and
ModelCheckpoint and trained all models multiple times.
To predict results of Regression model, we used RMSE, R2 score and Lift Chart. To predict results
of Classification model, we used Accuracy, Precision, Recall, F1-Score and Confusion Matrix.

## Experimental Results and analysis
We trained models with different parameters like activation functions and optimizers.
Following are the results of training both regression and classification models with different
combination of parameters of activation functions and optimizer.

## Regression model
| Activation\Optimizer | Adam | Sgd | Rmsprop |
| -------------------- |:----:|:---:|:-------:|
| |RMSE\R2 | RMSE\R2 | RMSE\R2 |
|Relu | 0.4558\0.7987 | 0.4450\0.8081 | 0.4485\0.8051 |
|Sigmoid | 0.4412\0.8114 | 0.4933\0.7642 | 0.4394\0.8129 |
|Tanh | 0.4414\0.8112 | 0.4887\0.8129 | 0.4624\0.8112 |

Table 1: Results of combinations of activation functions and optimizers in Neural Networks
Regression Model

## Classification model
| Activation\Optimizer | Adam | Sgd | Rmsprop |
| -------------------- |:----:|:---:|:-------:|
|Relu| 0.5338| 0.5415| 0.5365|
|Sigmoid| 0.5369| 0.1866| 0.5451|
|Tanh| 0.5393| 0.5389 |0.5405|

Table 2: Results of combinations of activation functions and optimizers in Neural Networks
Classification Model

According to above output from all models, it can be generalised that for regression model activation function=sigmoid and optimizer=rmsprop got best RMSE and for classification
activation function=sigmoid and optimizer=rmsprop got best accuracy.

Now, these result can be compared with models which were trained using sklearn.

|Regression Model|Using Sklearn| Using Neural network |
| ---------------|:-----------:|:--------------------:|
|RMSE\R2 | 0.30\0.72 | 0.4394\0.8129 |

|Classification model|Using Sklearn| Using Neural network |
| -------------------|:-----------:|:--------------------:|
|Accuracy | Logistic Reg-0.39, SVM-0.45, Nearest Neighbor-0.28, Multinomial Naive Bayes-0.32 | 0.5451 |

Table 3: Comparison of results of models using Sklearn and Neural Networks

Seeing the above results, it can be said that on this dataset Classification model using Neural
Networks provided good accuracy compared to all other classification models used, the Linear Regression model provided good RMSE score while Neural Network
Regression model provided good R2 score.

## Additional Feature
We used category as additional parameter to our input. A business can have multiple category. To
play with categories, we took first category from category list for all businesses. Furthermore, many
business has categories as null value so we removed that business. Then we performed one hot
encoding on categories. After this, we converted it into array and appended to tf-idf and review
count array input.

For categories, we trained model using activation function=sigmoid and optimizer=rmsprop for
Regression and activation function=sigmoid and optimizer=rmsprop for Classification, as these
combinations predicted best results.

## Analysis
Following shows the analysis of using additional feature:

|                          |Without using Categories | Using Categories |
| -------------------------|:-----------------------:|:----------------:|
|Regression Model - RMSE/R2| 0.4394/0.8129 | 0.4393/0.8120 |
|Classification model -Accuracy| 0.5451 | 0.5357 |

Table 4: Comparison of results of models without using Categories column and with using
Categories column

It can be seen, using categories to predict business stars doesn’t improve our results. The
importance of categories column in this model can also be found using Perturbation Feature
Ranking algorithm. Because of system configuration limit, we were not able to implement it on our
system with large dataset.
