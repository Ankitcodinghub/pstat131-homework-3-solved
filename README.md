# pstat131-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [PSTAT131 Homework 3 Solved](https://www.ankitcodinghub.com/product/pstat-131-homework-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;119130&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;PSTAT131 Homework 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
For this assignment, we will be working with part of a Kaggle data set that was the subject of a machine learning competition and is often used for practicing ML models. The goal is classification; specifically, to predict which passengers would survive the Titanic shipwreck.

Load the data from data/titanic.csv into R and familiarize yourself with the variables it contains using the codebook (data/titanic_codebook.txt).

Make sure you load the tidyverse and tidymodels!

Remember that you’ll need to set a seed at the beginning of the document to reproduce your results.

library(tidymodels) library(ISLR) # the Smarket data set library(ISLR2) # the Bikeshare data set library(discrim)

library(poissonreg) library(corrr) library(klaR) # naive bayes library(forcats) library(corrplot) library(pROC) tidymodels_prefer()

titanic &lt;- read.csv(“titanic.csv”) head(titanic)

## passenger_id survived pclass

## 1 1 No 3

## 2 2 Yes 1

## 3 3 Yes 3

## 4 4 Yes 1

## 5 5 No 3

## 6 6 No 3

## name sex age sib_sp parch

## 1 Braund, Mr. Owen Harris male 22 1 0

## 2 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female 38 1 0

## 3 Heikkinen, Miss. Laina female 26 0 0

## 5 Allen, Mr. William Henry male 35 0 0

## 6 Moran, Mr. James male NA 0 0

## ticket fare cabin embarked

## 1 A/5 21171 7.2500 &lt;NA&gt; S

## 2 PC 17599 71.2833 C85 C

## 3 STON/O2. 3101282 7.9250 &lt;NA&gt; S

## 4 113803 53.1000 C123 S

## 5 373450 8.0500 &lt;NA&gt; S

## 6 330877 8.4583 &lt;NA&gt; Q

Question 1

Split the data, stratifying on the outcome variable, survived. You should choose the proportions to split the data into. Verify that the training and testing data sets have the appropriate number of observations. Take a look at the training data and note any potential issues, such as missing data. Why is it a good idea to use stratified sampling for this data?

titanic$survived &lt;- as.factor(titanic$survived) titanic$survived &lt;- ordered(titanic$survived, levels = c(“Yes”, “No”)) titanic$pclass &lt;- as.factor(titanic$pclass)

titanic_split &lt;- initial_split(titanic, prop = 0.80, strata = survived) titanic_train &lt;- training(titanic_split) titanic_test &lt;- testing(titanic_split) head(titanic_train)

## passenger_id survived pclass name sex age sib_sp

## 1 1 No 3 Braund, Mr. Owen Harris male 22 1

## 6 6 No 3 Moran, Mr. James male NA 0

## 7 7 No 1 McCarthy, Mr. Timothy J male 54 0

## 8 8 No 3 Palsson, Master. Gosta Leonard male 2 3

## 13 13 No 3 Saundercock, Mr. William Henry male 20 0

## 14 14 No 3 Andersson, Mr. Anders Johan male 39 1

## parch ticket fare cabin embarked

## 1 0 A/5 21171 7.2500 &lt;NA&gt; S

## 6 0 330877 8.4583 &lt;NA&gt; Q

## 7 0 17463 51.8625 E46 S

## 8 1 349909 21.0750 &lt;NA&gt; S

## 13 0 A/5. 2151 8.0500 &lt;NA&gt; S

## 14 5 347082 31.2750 &lt;NA&gt; S

head(titanic_test)

## passenger_id survived pclass

## 5 5 No 3

## 9 9 Yes 3

## 28 28 No 1

## 39 39 No 3

## 49 49 No 3

## 50 50 No 3

## name sex age sib_sp parch

## 5 Allen, Mr. William Henry male 35 0 0

## 9 Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg) female 27 0 2

## 28 Fortune, Mr. Charles Alexander male 19 3 2

## 39 Vander Planke, Miss. Augusta Maria female 18 2 0

## 49 Samaan, Mr. Youssef male NA 2 0

## 50 Arnold-Franchi, Mrs. Josef (Josefine Franchi) female 18 1 0

## ticket fare cabin embarked

## 5 373450 8.0500 &lt;NA&gt; S

## 9 347742 11.1333 &lt;NA&gt; S

## 28 19950 263.0000 C23 C25 C27 S

## 39 345764 18.0000 &lt;NA&gt; S

## 49 2662 21.6792 &lt;NA&gt; C

## 50 349237 17.8000 &lt;NA&gt; S

Note that, there are some missing values in age, cabin. In addition, the value of ticket has different format. Because it can provide a more accurate representation of the population based on what’s used to divide it into different subsets. In our case we are looking to predict the survived people, so stratifying survived people that have different subsets will benefit our prediction.

Question 2

Using the training data set, explore/describe the distribution of the outcome variable survived.

titanic_train %&gt;%

ggplot(aes(x = survived,fill=survived)) + geom_bar() + ggtitle(“Count of Survived People”)

Count of Survived People

The distribution of the outcome is not even. The number of non-survived people is much more than the number of survived people.

Question 3

Using the training data set, create a correlation matrix of all continuous variables. Create a visualization of the matrix, and describe any patterns you see. Are any predictors correlated with each other? Which ones, and in which direction?

cor_titanic_train &lt;- titanic_train %&gt;%

select( -sex,-passenger_id, -name, -cabin, -ticket,-embarked,-survived) %&gt;% mutate(pclass = as.integer(pclass)) %&gt;% correlate(use = “pairwise.complete.obs”, method = “pearson”) rplot(cor_titanic_train)

In this plot, we first want to look for bright and large circles which show a strong correlation. Secondly, the size and shade depend on the absolute values of the coefficients, and the color depends on directions. – survived is positive correlated to sex, pclass.

– pclass is negative correlated to fare, age.

– age is negative correlated to sib_sp.

– sib_sp is positive correlated to parch.

– parch is negative correlated to sex, age.

Question 4

Using the training data, create a recipe predicting the outcome variable survived. Include the following predictors: ticket class, sex, age, number of siblings or spouses aboard, number of parents or children aboard, and passenger fare.

Recall that there were missing values for age. To deal with this, add an imputation step using step_impute_linear(). Next, use step_dummy() to dummy encode categorical predictors. Finally, include interactions between:

• Sex and passenger fare, and

• Age and passenger fare.

You’ll need to investigate the tidymodels documentation to find the appropriate step functions to use.

titanic_recipe &lt;- titanic_train %&gt;%

recipe(survived ~ pclass + sex + age + sib_sp + parch + fare) %&gt;% step_impute_linear(age) %&gt;% step_dummy(all_nominal_predictors()) %&gt;% step_interact(terms = ~ starts_with(“sex”):fare +

age:fare)

Question 5

Specify a logistic regression model for classification using the “glm” engine. Then create a workflow. Add your model and the appropriate recipe. Finally, use fit() to apply your workflow to the training data.

Hint: Make sure to store the results of fit(). You’ll need them later on.

log_reg &lt;- logistic_reg() %&gt;% set_engine(“glm”) %&gt;% set_mode(“classification”)

log_wkflow &lt;- workflow() %&gt;% add_model(log_reg) %&gt;% add_recipe(titanic_recipe) log_fit &lt;- fit(log_wkflow, titanic_train)

Question 6

Repeat Question 5, but this time specify a linear discriminant analysis model for classification using the “MASS” engine.

lda_mod &lt;- discrim_linear() %&gt;%

set_mode(“classification”) %&gt;% set_engine(“MASS”)

lda_wkflow &lt;- workflow() %&gt;% add_model(lda_mod) %&gt;% add_recipe(titanic_recipe) lda_fit &lt;- fit(lda_wkflow, titanic_train)

Question 7

Repeat Question 5, but this time specify a quadratic discriminant analysis model for classification using the “MASS” engine.

qda_mod &lt;- discrim_quad() %&gt;%

set_mode(“classification”) %&gt;% set_engine(“MASS”)

qda_wkflow &lt;- workflow() %&gt;%

add_model(qda_mod) %&gt;%

add_recipe(titanic_recipe) qda_fit &lt;- fit(qda_wkflow, titanic_train)

Question 8

Repeat Question 5, but this time specify a naive Bayes model for classification using the “klaR” engine. Set the usekernel argument to FALSE.

nb_mod &lt;- naive_Bayes() %&gt;%

set_mode(“classification”) %&gt;% set_engine(“klaR”) %&gt;% set_args(usekernel = FALSE)

nb_wkflow &lt;- workflow() %&gt;% add_model(nb_mod) %&gt;% add_recipe(titanic_recipe) nb_fit &lt;- fit(nb_wkflow, titanic_train)

Question 9

Now you’ve fit four different models to your training data.

Use predict() and bind_cols() to generate predictions using each of these 4 models and your training data. Then use the accuracy metric to assess the performance of each of the four models. Which model achieved the highest accuracy on the training data?

titanic_train_logistic &lt;- predict(log_fit, new_data = titanic_train, type = “prob”) log_acc &lt;- augment(log_fit, new_data = titanic_train)%&gt;%

accuracy(truth = survived, estimate = .pred_class)

titanic_train_lda &lt;- predict(lda_fit, new_data = titanic_train, type = “prob”) lda_acc &lt;- augment(lda_fit, new_data = titanic_train)%&gt;%

accuracy(truth = survived, estimate = .pred_class)

titanic_train_qda &lt;- predict(qda_fit, new_data = titanic_train, type = “prob”) qda_acc &lt;- augment(qda_fit, new_data = titanic_train)%&gt;%

accuracy(truth = survived, estimate = .pred_class)

titanic_train_nb &lt;- predict(nb_fit, new_data = titanic_train, type = “prob”) nb_acc &lt;- augment(nb_fit, new_data = titanic_train)%&gt;% accuracy(truth = survived, estimate = .pred_class)

titanic_train_predictions &lt;- bind_cols(titanic_train_logistic, titanic_train_lda,titanic_train_qda,titanic_train_nb)

titanic_train_predictions %&gt;% head()

## # A tibble: 6 x 8

## .pred_Yes…1 .pred_No…2 .pred_Yes~1 .pred~2 .pred~3 .pred~4 .pred~5 .pred~6

## &lt;dbl&gt; &lt;dbl&gt; &lt;dbl&gt; &lt;dbl&gt; &lt;dbl&gt; &lt;dbl&gt; &lt;dbl&gt; &lt;dbl&gt;

## 1 0.0949 0.905 0.0580 0.942 4.43e-3 0.996 1.20e-2 0.988

## 2 0.108 0.892 0.0627 0.937 4.27e-3 0.996 1.27e-2 0.987

## 3 0.279 0.721 0.231 0.769 3.98e-2 0.960 4.07e-1 0.593

## 4 0.0803 0.920 0.0583 0.942 3.09e-5 1.00 6.10e-5 1.00

## 5 0.166 0.834 0.0971 0.903 6.98e-3 0.993 1.42e-2 0.986

## 6 0.0167 0.983 0.0110 0.989 1.60e-3 0.998 7.59e-4 0.999

## # … with abbreviated variable names 1: .pred_Yes…3, 2: .pred_No…4,

## # 3: .pred_Yes…5, 4: .pred_No…6, 5: .pred_Yes…7, 6: .pred_No…8

accuracies &lt;- c(log_acc$.estimate, lda_acc$.estimate,

nb_acc$.estimate, qda_acc$.estimate)

models &lt;- c(“Logistic Regression”, “LDA”, “Naive Bayes”, “QDA”) results &lt;- tibble(accuracies = accuracies, models = models) results %&gt;% arrange(-accuracies)

## # A tibble: 4 x 2

## accuracies models

## &lt;dbl&gt; &lt;chr&gt;

## 1 0.813 Logistic Regression

## 2 0.796 LDA

## 3 0.774 QDA

## 4 0.768 Naive Bayes

The logistic regression model achieved the highest accuracy.

Question 10

Fit the model with the highest training accuracy to the testing data. Report the accuracy of the model on the testing data.

log_test &lt;- fit(log_wkflow, titanic_test) predict(log_test, new_data = titanic_test, type = “class”) %&gt;%

bind_cols(titanic_test %&gt;% select(survived)) %&gt;% accuracy(truth = survived, estimate = .pred_class)

## # A tibble: 1 x 3

## .metric .estimator .estimate

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt;

## 1 accuracy binary 0.827

Again using the testing data, create a confusion matrix and visualize it. Plot an ROC curve and calculate the area under it (AUC).

augment(log_test, new_data = titanic_test) %&gt;%

conf_mat(truth = survived, estimate = .pred_class) %&gt;% autoplot(type = “heatmap”)

52 14

17 96

Yes

No

Yes No

Truth

augment(log_test, new_data = titanic_test) %&gt;%

roc_curve(survived, .pred_Yes) %&gt;% autoplot()

# AUC augment(log_test, new_data = titanic_test) %&gt;%

roc_auc(survived, .pred_Yes)

## # A tibble: 1 x 3

## .metric .estimator .estimate

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt;

## 1 roc_auc binary 0.872

How did the model perform? Compare its training and testing accuracies. If the values differ, why do you think this is so?

The AUC is 0.8715 which indicates that the model performs well.

The accuracy of training is higher than the testing one because we optimized the training model.
