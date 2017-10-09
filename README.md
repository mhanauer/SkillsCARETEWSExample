---
title: "Early Warning Systems"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
This is an example of how to apply the CARET machine learning package in R to classify individuals or objects based upon covariates. I used an artificial data set that contains student’s ABC's (attendance, behavior, and course performance) as covariates and are used to predict whether or not a student will dropout (1 = dropout; 0 = non-dropout).  Therefore, I have three variables for each student: attendance rate, number of suspensions, and GPA. The goal is to develop an algorithm that uses current information on dropout and non-dropout students to predict whether or not future students will drop out.

This example is based upon the example provided by the creators of the CARET package who demonstrated a very similar process with a different data set. The example can be found here: http://topepo.github.io/caret/model-training-and-tuning.html
```{r, message=FALSE, warning=FALSE, echo=FALSE}
library(mlbench)
library(caret)
set.seed(12345)
# Dropouts
attendance = c(70:80)
supsensions = c(3:8)
gpa = seq(from = 1, to = 3.5, by = .1)
set.seed(12345)
datDrop = data.frame(attendance = sample(attendance, 100, replace = TRUE), supsensions = sample(supsensions, 100, replace = TRUE), gpa = sample(gpa, 100, replace= TRUE), dropout = rep(1, 100))

#Nondropouts
set.seed(12345)
attendance = c(75:100)
supsensions = c(0:5)
gpa = seq(from = 2.0, to = 4, by = .1)
set.seed(12345)
datNonDrop = data.frame(attendance = sample(attendance, 100, replace = TRUE), supsensions = sample(supsensions, 100, replace = TRUE), gpa = sample(gpa, 100, replace= TRUE), dropout = rep(0, 100))

dat = rbind(datDrop, datNonDrop); head(dat)
dat$dropout = as.factor(dat$dropout)
```
Next we need to partition the training sets from the testing sets. The createDataPartition in CARET does this by taking a stratified random sample of .75 of the data for training.

We then create both the training and testing data sets which will be used to develop and evaluate the model.
```{r}
inTrain = createDataPartition(y = dat$dropout, p = .75, list = FALSE)
training = dat[inTrain,]
testing = dat[-inTrain,] 
```
Here we are creating the cross validation method that will be used by CARET to create the training sets. Cross validation means to randomly split the data into k (in our case ten) data testing data sets and the repeated part just means to repeat this process k times (in our case ten as well).
```{r}
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10)
```
Now we are ready to the develop model. We use the train function in CARET to regress the dependent variable dropout onto all of the other covariates. Instead of explicitly naming all of the covariates, in the CARET package the “.” is used, which means include all of the other variables in the data set.

Next the method or type of regression is selected. Here we are using the gbm or Stochastic Gradient Boosting that is used for regression and classification. More information about the gbm package can be found here: https://cran.r-project.org/web/packages/gbm/gbm.pdf

The trControl is used to assign the validation method created above. It says run a gbm model with a ten cross validation method and repeat that process ten times. Finally, the verbose command just hides the calculations CARET computes for the user.
```{r}
set.seed(12345)
gbmFit1 <- train(dropout ~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl,
                 verbose = FALSE)
```
Let’s now inspect the results. The most important piece of information is the accuracy, because that is what CARET uses to choose the final model. It is the overall agreement rate between the cross validation methods. The Kappa is another statistical method used for assessing models with categorical variables such as ours.

CARET chose the first model with an interaction depth of 1, number of trees at 50, an accuracy of 95% and a Kappa of 90%.
```{r}
gbmFit1
```
Finally, we can use the training model to predict both classifications and probabilities for the test data set.

The first line of codes uses the built in predict function with the training model (gbmFit1) to predict values using the testing data set, which is the 25% of the data set that we set aside at the beginning of this example. We include the code “head” for your convenience so that R does not display the entire data set. If “head” were removed R would display all of the predictions.

The first piece of code includes the argument type = “prob”, which tells R to display the probabilities that a student is classified as non-dropout (0) or dropout (1). As we can see, there is a 98% probability that the first student in the data set is going to dropout.  

```{r}
predict(gbmFit1, newdata = head(testing), type = "prob")
```





