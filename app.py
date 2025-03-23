#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  
import plotly.express as px
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[3]:


file_path = 'forecasting_case_study.csv'
df = pd.read_csv(file_path)


# In[4]:


df.head()


# In[5]:


# Splitting data
X = df.drop(columns=['Sales', 'date'])  
y = df['Sales']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


# In[6]:


# Train & Predict
lr = LinearRegression()  
lr.fit(X_train, y_train)  
y_pred = lr.predict(X_test)


# In[7]:


# Evaluate
mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
print(f"Linear Regression - MAE: {mae}, MSE: {mse}")


# In[8]:


ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)  
y_pred_ridge = ridge.predict(X_test)


# In[9]:


rf = RandomForestRegressor(n_estimators=100, random_state=42)  
rf.fit(X_train, y_train)  
y_pred_rf = rf.predict(X_test)


# In[10]:


sarima = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1,1,1), seasonal_order=(1,1,1,12))  
sarima_fit = sarima.fit()  
df['SARIMA_Pred'] = sarima_fit.predict(start=len(df)-30, end=len(df)-1)


# In[11]:


df['SARIMA_Pred'] = sarima_fit.predict(start=len(df) - len(y_test), end=len(df) - 1)
df['SARIMA_Pred'].fillna(df['Sales'].mean(), inplace=True)  # Fill NaN with mean sales


# In[12]:


sarima_preds = df['SARIMA_Pred'].iloc[-len(y_test):]

if len(sarima_preds) != len(y_test):
    sarima_preds = sarima_preds[-len(y_test):]  # Ensure equal length


# In[13]:


# Evaluation Metrics Function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}


# In[14]:


# Store trained models in a dictionary
models = {
    "Ridge Regression": ridge,
    "Random Forest": rf
}


# In[24]:


# Store predictions in a dictionary
predictions = {
    "Ridge Regression": y_pred_ridge,
    "Random Forest": y_pred_rf
}


# In[15]:


# Collect Results
results = [
    evaluate_model(y_test, y_pred_ridge, "Ridge Regression"),
    evaluate_model(y_test, y_pred_rf, "Random Forest"),
    evaluate_model(y_test, df['SARIMA_Pred'].iloc[-len(y_test):], "SARIMA")
]


# In[16]:


# Convert results into DataFrame
results = pd.DataFrame(results)
print(results)


# In[17]:


# Visualize Model Performance
plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="RMSE", data=results)
plt.title("Model Performance (Lower RMSE is Better)")
plt.show()


# In[18]:


# Identify the Best Model (Lowest RMSE)
best_model_name = results.loc[results["RMSE"].idxmin(), "Model"]


# In[19]:


# Streamlit Dashboard
st.title("📊 Sales Forecasting Dashboard")


# In[20]:


# Model Performance Table
st.subheader("📈 Model Performance Metrics")
st.dataframe(results)


# In[21]:


# Model Performance Visualization
st.subheader("🔍 Model Comparison (RMSE)")
fig_bar = px.bar(results, x="Model", y="RMSE", title="Model Performance (Lower RMSE is Better)")
st.plotly_chart(fig_bar)


# In[22]:


# Feature Importance for Random Forest
st.subheader("🎯 Feature Importance - Random Forest")
feature_importance = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False)
fig_importance = px.bar(x=feature_importance.index, y=feature_importance.values, title="Feature Importance")
st.plotly_chart(fig_importance)


# In[25]:


# Actual vs. Predicted Sales Visualization
st.subheader("📊 Actual vs Predicted Sales")
pred_df = pd.DataFrame({
    "date": df['date'].iloc[-len(y_test):],
    "Actual Sales": y_test.values,
    "Ridge Regression": predictions["Ridge Regression"],
    "Random Forest": predictions["Random Forest"],
    "SARIMA": df['SARIMA_Pred'].iloc[-len(y_test):]
})


# In[26]:


fig_line = px.line(pred_df, x="date", y=["Actual Sales", "Ridge Regression", "Random Forest", "SARIMA"],
                   title="Actual vs Predicted Sales")
st.plotly_chart(fig_line)


# In[27]:


# Future Forecasting Using Best Model
st.subheader(f"🚀 Future Sales Forecast (Using {best_model_name})")

if best_model_name != "SARIMA":
    best_model = models[best_model_name]
    best_model.fit(X, y)
    future_X = X.tail(30)
    future_dates = pd.date_range(start=df['date'].max(), periods=30)
    future_preds = best_model.predict(future_X)
else:
    future_dates = pd.date_range(start=df['date'].max(), periods=30)
    future_preds = sarima_fit.forecast(steps=30)


# In[28]:


future_df = pd.DataFrame({"date": future_dates, "Predicted Sales": future_preds})
fig_future = px.line(future_df, x="date", y="Predicted Sales", title="Future Sales Forecast")
st.plotly_chart(fig_future)

