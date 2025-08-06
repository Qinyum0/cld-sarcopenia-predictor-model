#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator

# 页面配置
st.set_page_config(
 page_title="Sarcopenia Risk Prediction in CLD Patients",
 layout="wide",
 initial_sidebar_state="expanded"
)

class XGBCompatWrapper(BaseEstimator):
 def __init__(self, model):
     self.model = model
     
 def predict(self, X):
     return self.model.predict(X)
     
 def predict_proba(self, X):
     return self.model.predict_proba(X)
     
# 加载预训练模型
try:
 best_xgb_model = joblib.load("cld_model.pkl")
 # 应用兼容性包装器
 if hasattr(best_xgb_model, 'use_label_encoder'):
     best_xgb_model = XGBCompatWrapper(best_xgb_model)
except Exception as e:
 st.error(f"Failed to load model: {str(e)}")
 st.stop()

def predict_prevalence(patient_data):
 """使用预训练模型进行预测"""
 try:
     input_df = pd.DataFrame([patient_data])
     proba = best_xgb_model.predict_proba(input_df)[0]
     prediction = best_xgb_model.predict(input_df)[0]
     return prediction, proba
 except Exception as e:
     st.error(f"Prediction error: {str(e)}")
     return None, None

def main():
 st.title('Sarcopenia Risk Prediction in CLD Patients')
 st.markdown("""
 This tool predicts sarcopenia risk in chronic lung disease (CLD) patients.
 """)
 
 # 侧边栏输入
 st.sidebar.header('Patient Parameters')
 age = st.sidebar.slider('Age', 45, 100, 50)
 gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
 residence = st.sidebar.selectbox('Residence', ['Urban', 'Rural'])
 waist = st.sidebar.slider('Waist Circumference', 15, 150, 60)
 
 if st.sidebar.button('Predict'):
     patient_data = {
         'age': age,
         'gender': 0 if gender == 'Female' else 1,
         'residence': 0 if residence == 'Urban' else 1,
         'waist': waist
     }
     
     prediction, proba = predict_prevalence(patient_data)
     
     if prediction is not None:
         st.subheader('Prediction Results')
         col1, col2 = st.columns(2)
         
         with col1:
             if prediction == 1:
                 st.error('**High Risk** of Sarcopenia')
             else:
                 st.success('**Low Risk** of Sarcopenia')
         
         with col2:
             risk_percent = proba[1]*100 if prediction == 1 else proba[0]*100
             st.metric("Probability", f"{risk_percent:.1f}%")
         
         st.progress(proba[1])
         st.write(f"**Probability Distribution:**")
         st.write(f"- No Sarcopenia: {proba[0]*100:.2f}%")
         st.write(f"- Sarcopenia: {proba[1]*100:.2f}%")

if __name__ == '__main__':
 main()


# In[2]:





# In[ ]:






