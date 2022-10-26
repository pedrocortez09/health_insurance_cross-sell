# <p align="center" dir="auto">Health Insurance - Cross Sale</p>
## <p align="center" dir="auto">Learn to Rank - Order a List of Clients</p>
<div align="center">
<img src="https://user-images.githubusercontent.com/103151311/197250556-4d5312fa-1d2e-407e-a5a2-34bd0049db6f.png" width="700" />
</div>


# 1. Context
This project is based in a Kaggle Challenge wich simulates a business problem. An insurance company wants to start a cross-sell strategy. 
They want to sell car insurance to people who already is their client with a base of 381.109 clients. They made a poll to investigate possible customers.


# 2. Business Problem and Plan Solution

## 2.1. Problem
The business team have a budget to contact 20.000 people and they want to manage to sell the new product for the maximum clients possible,
but they have and order to contact this 381.109 clients.

+ Limited calls to contact clients;
+ Current method is random calls.

## 2.2. Solution

+ Use a Machine Learning Model to get a propensity score of each client (Learn to rank);
+ Propensity score will be accessible through Google Sheets API.

## 2.3. Business Assumptions

The Marketing Manager stated that they are only able to offer adequate treatment during sales campaign - including calls, invitations for conversations 
in person and special attention to clarify doubts - up to 40% of last year's health insurance policyholders. Therefore, 
the company wants to obtain an ordered list of customers most likely to purchase vehicle insurance to maximize customer conversion.


# 3. Solution Plan

When each customer signed up for health insurance during last year, they also completed a survey with relevant data to a car insurance decision-making, 
such as ownership of a driver's license, vehicle age, wheter the customer got his/ her vehicle damaged in the past, among other information.

The result of this survey will guide the creation of a machine learning model that will produce an ordered list of customers most likely to acquire a car insurance.


## 3.1 Steps to deliver a solution:
+ **Data Description**: Treatment of the original data like: fill NA's, change data types and descriptive statistical analyses;
+ **Feature Engineering**: Create new derived features from the original ones to make a better Machine Learning Model;
+ **Data Filtering**: Select lines and columns to drop from the data like days when the stores were closed;
+ **Exploration Data Analyses**: Analyses of the features distribution to take insights of how impactant each feature will be for our model;
+ **Data Preparation**: Reescale each feature to a commom range for the model to learn;
+ **Feature Selection**: Use an algorithm (Boruta) to select the best features for the model;
+ **Machine Learning Modelling**: Apply Machine Learning models and measure their perfomance based on Precision at K and Recall at K measures;
+ **Hyper Parameter Fine Tuning**: Choose randomly the bests parameters for our chosen model;
+ **Business Results**: A Google Sheets spreadsheet with a button to apply each customer's score



# 4. Top 3 Data Insights
## **1 - Customers who have already crashed their car have a higher conversion rate**

Vehicle Damage   | Interested in vehicle insurance | Not interested in vehicle insurance
:--------- | :------ | :------
Yes   | 23,77% | 76,23%
No | 	0,52% | 99,48%

<div align="left">
<img src="https://user-images.githubusercontent.com/103151311/197256957-56958a30-35d4-47a5-bb56-57ff46e03103.png" width="700" />
</div>
  
## **2 - Customers with older cars have a higher conversion rate**

Vehicle Age   | Interested in vehicle insurance | Not interested in vehicle insurance
:--------- | :------ | :------
Below 1 year | 4,37% | 95,63%
Between 1 and 2 years | 	17,38% | 82,62%
Above 2 years | 	29,37% | 70,63%

<div align="left">
<img src="https://user-images.githubusercontent.com/103151311/197258245-4dba099d-65b5-4b09-bb9f-1e5599e69956.png" width="700" />
</div>

## **3 - Age is relevant in the decision to take out vehicle insurance**

<div align="left">
<img src="https://user-images.githubusercontent.com/103151311/197259980-d2ff1cd6-4a57-4cd7-96f6-8369c59076f4.png" width="700" />
</div>


# 5. Machine Learning Results
All machine learning algorithms were trained using cross validation on training data. 
the metrics to measure the performance of each model were: Precision at K and Recall at K

**Models Used:**
+ K-Nearest Neighbors;
+ Logistic Regression;
+ Extra Trees;
+ Random Forest;
+ XGBoost;
+ Light Gradient Boosting Machine.

**Models Performance:**

<div align="left">
<img src="https://user-images.githubusercontent.com/103151311/197261352-ed375e55-e3b1-42e5-836e-959d4a9a12a2.png" width="700" />
</div>

## 5.1 Model Selected
**The model selected was Light Gradient Boosting Machine, which gave this grafics for the results:**

<div align="left">
<img src="https://user-images.githubusercontent.com/103151311/197261857-9e1803c3-1d16-4ab2-9b3b-d592f7540b6c.png" width="700" />
</div>


## 5.2 Results:
+ On the Cumulative Gain curve, where the Y-axis is the percentage of all my interested clients, and X-axis is the percentage of my database tells us that if
 we call to 40% of our base, we will reach about 92% of all my interested clients.
 + On the Lift curve, where the Y-axis represents how much better my model is than a random model, and X-axis is the percetage of database.

# 6. Business Results
We can note that by applying this model approximately 92% of potential customers 
interested in vehicular insurance will be converted by addressing 40% of the total customers currently with only health insurance.

The model was deployed on Heroku (https://www.heroku.com/) and available through a spreadsheet
in Google Sheets (https://docs.google.com/spreadsheets/d/14gacR_DBBkWEnvGO4R5nIf2C4ou7c8pqsrSLTrb-ogE/edit#gid=0)

Any employee of the insurance company can use the spreadsheet and establish a ranking of customers most likely to purchase vehicle insurance, 
with direct production data.

As can be seen in the demonstration below, there is a button that, once activated, after a few seconds, returns the list already sorted by the customers 
most likely to purchase the new product.

https://user-images.githubusercontent.com/103151311/197268583-6722a632-aec4-4ac0-8ea8-3f6d04d439c9.mp4


# 7. Conclusions

With this model, it is expected that the company's profit will increase due to the decrease in the cost of random calls,
and also the customer conversion rate will be higher.

the sales department will have a very easy access through google sheets to decide the order of their calls.

# 8. Next Steps

For future learning to rank problems or new CRISP rounds for this business scenario, consider training these models:

+ Balanced Bagging Classifier;
+ Easy Ensemble Classifier;
+ Random Under Sampler Classifier;

