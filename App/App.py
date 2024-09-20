import streamlit as st
import pickle
import pandas as pd
import numpy as np

from utils.utils import CreateAdditionalAttributes

st.set_page_config(page_title='House Prices Predictor', layout = 'wide', page_icon='🏠')


@st.cache_resource
def load_resource(model_path, pipeline_path):
    model = pickle.load(open(model_path, 'rb'))
    pipeline = pickle.load(open(pipeline_path, 'rb'))
    
    return model, pipeline

@st.cache_data
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

data_path = "../data/housing.csv"
data = load_data(data_path)

model_path, pipeline_path = '../best_model.pkl', '../full_pipeline.pkl'

model, pipeline = load_resource(model_path, pipeline_path)


left_col, mid_col, right_col = st.columns([1,2,1])

with mid_col:    
    st.title('California House Prices Predictor🏠',)
    longitude = st.number_input(
        label='Longitude', 
        min_value=int(np.min(data.longitude.values)), 
        max_value=int(np.max(data.longitude.values)))
    
    latitude = st.number_input(
        label='Latitude', 
        min_value=int(np.min(data.latitude.values)), 
        max_value=int(np.max(data.latitude.values)))
    
    housing_median_age = st.number_input(
        label='Housing Median Age', 
        min_value=int(np.min(data.housing_median_age.values)), 
        max_value=int(np.max(data.housing_median_age.values)))
    
    total_rooms = st.number_input(
        label='Total Rooms', 
        min_value=1.0, 
        max_value=float(np.max(data.total_rooms.values)))
    
    total_bedrooms = st.number_input(
        label='Total Bedrooms', 
        min_value=1.0, 
        max_value=float(np.max(data.total_bedrooms.values)))
    
    population = st.number_input(
        label='Population', 
        min_value=float(np.min(data.population.values)), 
        max_value=float(np.max(data.population.values)))
    
    households = st.number_input(
        label='Households', 
        min_value=float(np.min(data.households.values)), 
        max_value=float(np.max(data.households.values)))
    
    median_income = st.number_input(
        label='Median Income', 
        min_value=float(np.min(data.median_income.values))*100000, 
        max_value=float(np.max(data.median_income.values))*10000),
    
    ocean_proximity = st.selectbox('Ocean Proximity', options=data.ocean_proximity.unique())
    
    button = st.button(label='Get Your Median House Value')
    
if button:
    full_features = pd.DataFrame({
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }, index=[0])
    ready_features = pipeline.transform(full_features)
    predicted_value = model.predict(ready_features)
    
    st.write(f'The Approximate Median House Value -> {predicted_value[0]}')