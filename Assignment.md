Prediction Assignment
================

Introduction
------------

This project will review data captured from exercise devices of individuals performing a barbell exercise 5 different ways. Using machine learning techniques on a training set, a prediction model will be generated and then evaluated against a hold out testing set.

Special thanks to the groups publishing their data and making it available to the public: <http://groupware.les.inf.puc-rio.br/har>

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

The original training dataset can be found here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The original testing dataset can be found here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The Data
--------

After loading the data, a quick examination shows 160 variables. These capture everything from the name of the participant and timestamp to the pitch, roll and yaw of the exercise being done on X, Y, Z axises. Finally the data is divided into 5 classes (A thru E) to denote the type of motion used to complete the barbell lift.

``` r
trainingRaw <- read.csv("pml-training.csv")
datatoTest <- read.csv("pml-testing.csv")
dim(trainingRaw)
```

    ## [1] 19622   160

Not all of the 160 variables will be useful to the analysis. Many are blank or have near zero variance (NZV).

Processing and Cleaning the Data
--------------------------------

To begin, the training data set will be divided into a training subset and a test subset with a 70/30 ratio. Then reduced to the variables most likely to be the best predictors of the exercise. NZV and variables predominantly N/A are removed.

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.4

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.4.3

``` r
set.seed(42918)
partition <- createDataPartition(trainingRaw$classe, p = 0.7, list = FALSE)
training <- trainingRaw[partition, ]
testing <- trainingRaw[-partition, ]

noVar <- nearZeroVar(trainingRaw)

training <- training[, -noVar]
testing <- testing [, -noVar]

isNA <- sapply(training, function(x) mean(is.na(x))) > .95
training <- training[, isNA==FALSE]
testing <- testing[, isNA==FALSE]

# Columns 1 thru 5 represent timestamp and user names. 
training <- training[, -(1:5)]
testing <- testing[, -(1:5)]

dim(training)
```

    ## [1] 13737    54

The dataset has been reduced to 54 variables after cleaning and processing.

Training a Prediction Model
---------------------------

To start, an exploration using a random forest technique and a generalized boosted model will be used to establish a basline of accuracy.

GBM:

``` r
set.seed(54321)

training$classe = as.factor(training$classe)
fit2 <- train(classe ~ . , method = "gbm" , verbose = FALSE, data = training)
fit2p <- predict(fit2, testing)
confusionMatrix(fit2p, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671   15    0    1    0
    ##          B    2 1113    9    3    2
    ##          C    0   11 1012   11    2
    ##          D    1    0    5  949   18
    ##          E    0    0    0    0 1060
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9864          
    ##                  95% CI : (0.9831, 0.9892)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9828          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9772   0.9864   0.9844   0.9797
    ## Specificity            0.9962   0.9966   0.9951   0.9951   1.0000
    ## Pos Pred Value         0.9905   0.9858   0.9768   0.9753   1.0000
    ## Neg Pred Value         0.9993   0.9945   0.9971   0.9969   0.9954
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1891   0.1720   0.1613   0.1801
    ## Detection Prevalence   0.2867   0.1918   0.1760   0.1653   0.1801
    ## Balanced Accuracy      0.9972   0.9869   0.9907   0.9898   0.9898

The GBM model has a 98.8% accuracy, with 1.2% out of sample error.

``` r
trC <- trainControl( method = "cv", number = 3)
fit1 <- train(classe ~ . , method = "rf", trainControl = trC, data = training, ntree = 200)

fit1p <- predict(fit1, testing)
confusionMatrix(fit1p, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    6    0    0    0
    ##          B    0 1131    2    0    0
    ##          C    0    2 1024    3    0
    ##          D    0    0    0  961    4
    ##          E    0    0    0    0 1078
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9971          
    ##                  95% CI : (0.9954, 0.9983)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9963          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9930   0.9981   0.9969   0.9963
    ## Specificity            0.9986   0.9996   0.9990   0.9992   1.0000
    ## Pos Pred Value         0.9964   0.9982   0.9951   0.9959   1.0000
    ## Neg Pred Value         1.0000   0.9983   0.9996   0.9994   0.9992
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1922   0.1740   0.1633   0.1832
    ## Detection Prevalence   0.2855   0.1925   0.1749   0.1640   0.1832
    ## Balanced Accuracy      0.9993   0.9963   0.9985   0.9980   0.9982

The RandomForest model has a 99.78% accuracy, with a .22% out of sample error.

Conclusion
----------

With an accuracy of over 99%, the Random Forest method will be applied to the 20 records in the original pml-testing dataset. Additional methods, or even stacking techniques, could be explored; however, the high accuracy rate does not warrant it for the purposes of this project.
