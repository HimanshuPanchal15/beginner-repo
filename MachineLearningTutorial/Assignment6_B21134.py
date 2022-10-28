#QUESTION-1 
#Importing required libraries and modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import math
from statsmodels.tsa.ar_model import AutoReg


#Reading the given csv file  
df = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')

#PART-(a)
#Plotting line plot of Covid-19 cases
plt.plot(df)
plt.xlabel('Year-Month', fontsize=14)
plt.xticks(rotation = 45)
plt.ylabel('New Confirmed cases', fontsize=14)
plt.title('Line Plot Q1-(a)', fontsize=16)
plt.show()

#PART-(b)
#Listing the different lag values to generate multiple time sequences
lag = [1,2,3,4,5,6]
#An empty list to append the autocorrelation values
c = []

#Defining a function to calculate Pearson correlation (Autocorrelation)
def Corr(n):
    #For lag-p sequence, removing the first p elements from the dataframe
    df_top = df.iloc[n:]
    #Removing the last p elements from the dataframe
    df_bottom = df.iloc[:len(df)-n] 
    
    #PART-(c)
    #Scatter plot between the given time sequence and lag-1 time sequence 
    if n == 1:
        plt.scatter(df_top['new_cases'], df_bottom['new_cases'])
        plt.xlabel('Given time sequence', fontsize=14)
        plt.ylabel('Lag-1 time sequence', fontsize=14)
        plt.title('Scatter plot between the given time sequence and lag-1 sequence', fontsize=16)
        plt.show()
        
    #Calculating Pearson correlation (Autocorrelation)        
    corr = np.corrcoef(df_top['new_cases'], df_bottom['new_cases'])
    c.append(corr[0][1])
    print(f'Autocorrelation coefficient for the Lag-{n} sequence is',corr[0][1])
        
for i in lag:
    Corr(i)

#PART-(d)
#Plotting line plot between autocorrelation coefficients and lagged values    
plt.plot(lag,c)    
plt.xlabel('Lagged values', fontsize=14)
plt.ylabel('Autocorrelation coefficient', fontsize=14)
plt.title('Line plot between autocorrelation coefficients and lagged values', fontsize=16)
plt.show()

#PART-(e)
#Ploting Correlogram or Auto Correlation Function
plot_acf(df, lags=lag)
plt.title('Correlogram', fontsize=16)
plt.show()

#QUESTION 2
#PART-(a)
#Splitting the data into train and test data
#The initial 65% of the sequence for training data and the remaining 35% of the sequence as test data.
test_size = 0.35  
X = df.values
tst_sze = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_sze],X[len(X)-tst_sze:]

#Plotting line plot of train data
plt.plot(train)
plt.ylabel('New Confirmed cases', fontsize=14)
plt.title('Line Plot of train data set', fontsize=16)
plt.show()

#Plotting line plot of test data
plt.plot(test)
plt.ylabel('New Confirmed cases', fontsize=14)
plt.title('Line Plot of test data set', fontsize=16)
plt.show()

#AutoRegression(AR) model
#AR(5) that is lag=5
model = AutoReg(train,lags=5)
#Fit/train the model
model_fit = model.fit()  

#Coefficients of AR(5) model
coef = model_fit.params   
print('\nThe coefficients of AR(5) model are [W0  W1  W2  W3  W4  W5] =',coef)

#PART-(b)
#Listing the last 5 elements of train data
history = train[len(train)-5:]
history = [history[i] for i in range(len(history))]

#An empty list reqired to append the predicted values
Xt_pred =[]

for i in range(len(test)):
    length = len(history)
    lag = [history[j] for j in range(length-5,length)]
    
    #Initializing to W0
    y_hat = coef[0] 
    for d in range(5):
        #y_hat = W0 + W1*X(t-1) + .... + Wp*X(t-p)
        y_hat += coef[d+1]*lag[5-d-1]
    Xt_pred.append(y_hat)
    
    #Appending the test values to the list of last 5 elements of train data
    obs = test[i]
    history.append(obs)
    
#PART-(b(i))
#Plotting Scatter plot between actual and predicted test values
plt.scatter(test, Xt_pred)
plt.xlabel('Actual values', fontsize=14)
plt.ylabel('Predicted values', fontsize=14)
plt.title('Scatter plot showing actual and predicted test values', fontsize=18)
plt.show()

#PART-(b(ii))
#Plotting Line plot showing actual and predicted test values.
plt.plot(test)
plt.plot(Xt_pred, color='red')
plt.ylabel('Covid-19 cases', fontsize=14)
plt.title('Line plot showing actual and predicted test values', fontsize=18)
plt.show()

#PART-(b(iii))
#Computing RMSE (%) and MAPE between actual and predicted test data
#RMSE(%) = (sqrt(sum((actual value - predicted value)**)/no. of test samples)/mean of test values)*100
N = np.size(test)
r = 0    
for i in range(N):
    r += (Xt_pred[i] - test[i])**2     
rmse = math.sqrt(r/N)
print('\nRMSE(%) between actual and predicted values of test data is',(rmse/test.mean())*100)

#MAPE = (sum(mod(actual value - predicted value)/actual value)/no.of test saples)*100
m = 0
for i in range(len(test)):
    m += float(abs(test[i] - Xt_pred[i])/test[i])
mape = (m/N)*100       
print('\nMAPE between actual and predicted values of test data is',mape)

#Question 3
#Splitting the data into train and test data
test_size = 0.35  #35% test
X = df.values
tst_sze = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_sze],X[len(X)-tst_sze:]

#Listing the p values to generate lag-p time sequences
p = [1,5,10,15,25]

#Empty lists to store the calculated RMSE and MAPE values
RMSE = []
MAPE =[]

#Defining a function to generate AutoRegression(AR) model and calculate RMSE and MAPE
def AR_Error(k):
    model = AutoReg(train,lags=k)
    #Fit train model
    model_fit = model.fit()
    #Coefficients of the AR(k) model
    coef = model_fit.params
    
    #listing the last k elements from train data
    history = train[len(train)-k:]
    history = [history[i] for i in range(len(history))]
    
    #An empty list to store the predicted values
    Xt_pred =[]
    
    for i in range(len(test)):
        length = len(history)
        lag = [history[j] for j in range(length-k,length)]
        
        #Initializing to W0
        y_hat= coef[0]
        for d in range(k):
            #y_hat = W0 + W1*X(t-1) + .... + Wk*X(t-k)
            y_hat += coef[d+1]*lag[k-d-1]   
        Xt_pred.append(y_hat)
        
        #Appending the test values to the list of last k elements of train data
        obs = test[i]
        history.append(obs)

    #Computing RMSE (%) and MAPE between actual and predicted test data
    #RMSE(%) = (sqrt(sum((actual value - predicted value)**)/no. of test samples)/mean of test values)*100
    N = np.size(test)
    r = 0    
    for i in range(N):
        r += (Xt_pred[i] - test[i])**2     
    rmse = math.sqrt(r/N)
    RMSE.append((rmse/test.mean())*100)

    #MAPE = (sum(mod(actual value - predicted value)/actual value)/no.of test saples)*100
    m = 0
    for i in range(len(test)):
        m += float(abs(test[i] - Xt_pred[i])/test[i])
    mape = (m/N)*100       
    MAPE.append(mape)

for i in p:
    AR_Error(i)
    
#Creating a dataframe to store the lag values and their respective error values 
p_error = pd.DataFrame()
p_error['Lag'] = p
p_error['RMSE'] = RMSE
p_error['MAPE'] = MAPE
print(p_error)

#Plotting a bar chart of RMSE values of respective lag values
plt.bar(p,RMSE)
plt.xlabel('Lag values', fontsize=14)
plt.ylabel('RMSE(%)', fontsize=14)
plt.title('Bar chart showing RMSE values', fontsize=18)
plt.show()

#Plotting a bar chart of MAPE values of respective lag values
plt.bar(p,MAPE)
plt.xlabel('Lag values', fontsize=14)
plt.ylabel('MAPE', fontsize=14)
plt.title('Bar chart showing MAPE values', fontsize=18)
plt.show()

#QUESTION 4
#Splitting the data into train and test data
test_size = 0.35  #35% test
tst_sze = math.ceil(len(df)*test_size)
train,test = df[:len(df)-tst_sze],df[len(df)-tst_sze:]
train_list = list(train['new_cases'])
test_list = list(test['new_cases'])

#Listing the different lag values to generate multiple time sequences
lag = list(range(1,len(train)+1))
#An empty list to append the autocorrelation values
c = []

#Defining a function to calculate Pearson correlation (Autocorrelation)
def Corr(n):
    #For lag-p sequence, removing the first p elements from the dataframe
    train_top = train.iloc[n:]
    #Removing the last p elements from the dataframe
    train_bottom = train.iloc[:len(train)-n] 
    
    #Calculating Pearson correlation (Autocorrelation)        
    corr = np.corrcoef(train_top['new_cases'], train_bottom['new_cases'])
    c.append(corr[0][1])

#Calculating autocorrelation for all the lag values      
for i in lag:    
    Corr(i)
    
#An empty list to store the lag values which satisfy the condition that Autocorrelation > 2/sqrt(no. of train samples)    
k = []
for j in range(len(c))    :
    if c[j] > (2/math.sqrt(len(train))):
        k.append(lag[j])
        
print('The lag values which satisfy the given condition are:')
print(k)
print()
print('As the series of lag values satisfying the condition break at 77 so the Optimal no. of lags are: 77')
  
#Defining a function to generate AutoRegression(AR) model and calculate RMSE and MAPE
def AR_Error(k):
    model = AutoReg(train_list,lags=k)
    #Fit train model
    model_fit = model.fit()
    #Coefficients of the AR(k) model
    coef = model_fit.params
    
    #listing the last k elements from train data
    history = train_list[len(train_list)-k:]
    history = [history[i] for i in range(len(history))]
    
    #An empty list to store the predicted values
    Xt_pred =[]
    
    for i in range(len(test_list)):
        length = len(history)
        lag = [history[j] for j in range(length-k,length)]
        
        #Initializing to W0
        y_hat= coef[0]
        for d in range(k):
            #y_hat = W0 + W1*X(t-1) + .... + Wk*X(t-k)
            y_hat += coef[d+1]*lag[k-d-1]   
        Xt_pred.append(y_hat)
        
        #Appending the test values to the list of last k elements of train data
        obs = test_list[i]
        history.append(obs)

    #Computing RMSE (%) and MAPE between actual and predicted test data
    #RMSE(%) = (sqrt(sum((actual value - predicted value)**)/no. of test samples)/mean of test values)*100
    N = np.size(test_list)
    r = 0    
    for i in range(N):
        r += (Xt_pred[i] - test_list[i])**2     
    rmse = math.sqrt(r/N)
    print(f'RMSE(%) value when lag = {k} is',(rmse/np.mean(test_list))*100)

    #MAPE = (sum(mod(actual value - predicted value)/actual value)/no.of test saples)*100
    m = 0
    for i in range(len(test_list)):
        m += float(abs(test_list[i] - Xt_pred[i])/test_list[i])
    mape = (m/N)*100 
    print(f'MAPE value when lag = {k} is', mape)
    
AR_Error(77)