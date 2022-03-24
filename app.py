import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder,labelencoder
from load_model import get_model

model = get_model(model_path = r'model/randomforestmodel.pkl')


#model = joblib.load(r'model/randomforestmodel.pkl')

st.set_page_config(page_title="Site Energy Utilization Index App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_state_factor = ['State_1', "State_2", "State_4", "State_6", "State_8",
                        "State_10", "State_11"]

options_building_class = ['Commercial', 'Residential']

options_facility_type = ['Grocery_store_or_food_market',
 'Warehouse_Distribution_or_Shipping_center', 'Retail_Enclosed_mall',
 'Education_Other_classroom', 'Warehouse_Nonrefrigerated',
 'Warehouse_Selfstorage', 'Office_Uncategorized' 'Data_Center',
 'Commercial_Other', 'Mixed_Use_Predominantly_Commercial',
 'Office_Medical_non_diagnostic', 'Education_College_or_university',
 'Industrial', 'Laboratory', 'Public_Assembly_Entertainment_culture',
 'Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized',
 'Lodging_Hotel', 'Retail_Strip_shopping_mall' 'Education_Uncategorized',
 'Health_Care_Inpatient', 'Public_Assembly_Drama_theater',
 'Public_Assembly_Social_meeting', 'Religious_worship',
 'Mixed_Use_Commercial_and_Residential', 'Office_Bank_or_other_financial',
 'Parking_Garage', 'Commercial_Unknown',
 'Service_Vehicle_service_repair_shop', 'Service_Drycleaning_or_Laundry',
 'Public_Assembly_Recreation', 'Service_Uncategorized',
 'Warehouse_Refrigerated', 'Food_Service_Uncategorized',
 'Health_Care_Uncategorized', 'Food_Service_Other',
 'Public_Assembly_Movie_Theater', 'Food_Service_Restaurant_or_cafeteria',
 'Food_Sales','Public_Assembly_Uncategorized' 'Nursing_Home',
 'Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare',
 '5plus_Unit_Building', 'Multifamily_Uncategorized',
 'Lodging_Dormitory_or_fraternity_sorority', 'Public_Assembly_Library',
 'Public_Safety_Uncategorized', 'Public_Safety_Fire_or_police_station',
 'Office_Mixed_use', 'Public_Assembly_Other', 'Public_Safety_Penitentiary',
 'Health_Care_Outpatient_Uncategorized', 'Lodging_Other',
 'Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse',
 'Public_Assembly_Stadium', 'Lodging_Uncategorized', '2to4_Unit_Building',
 'Warehouse_Uncategorized']
       

features = ['mintempjandec','state_factor','maxtempjandec','building_class',
            'max_wind_speed','facility_type','floor_area','elevation',
            'cooling_degree_days','heating_degree_days']


st.markdown("<h1 style='text-align: center;'>Site Energy Utilization Index App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        mintempjandec = st.slider("Min Temp in Fahrenheit from Jan to Dec: ", 16.25, 55.75, value=0., step = 10., format="%f")
        state_factor = st.selectbox("Select State Factor: ", options=options_state_factor)
        maxtempjandec = st.slider("Max Temp in Fahrenheit from Jan to Dec: ", 66, 1001, value=0, step = 50, format="%d")
        building_class = st.selectbox("Select Building  Class: ", options=options_building_class)
        max_wind_speed = st.slider("Max wind speed : ", 1, 23, value=0,step = 1, format="%d")
        facility_type = st.selectbox("Select Facility Type: ", options=options_facility_type)
        floor_area = st.slider("Specify the floor area : ", 943, 6385382, value=0, step = 50, format="%d")
        elevation = st.slider("Specify the elevation : ", -6.4, 1924.50, value=0.,step = 10., format="%f")
        cooling_degree_days = st.slider("Cooling  degree in Fahrenheit : ", 0, 412, value=0, step = 10, format="%d")
        heating_degree_days = st.slider("Heating degree in Fahrenheit: ", 33.1, 660.10, value=0., step = 10., format="%f")
        

        
        submit = st.form_submit_button("Predict")


    if submit:
        state_factor = ordinal_encoder(state_factor, options_state_factor)
        #building_class = ordinal_encoder(building_class, options_building_class)
        building_class = labelencoder(building_class, options_building_class)
        #facility_type =  ordinal_encoder(facility_type, facility_type)
        facility_type = labelencoder(facility_type, options_facility_type)
        

        data = np.array([mintempjandec,state_factor,maxtempjandec,building_class,
                         max_wind_speed, facility_type,floor_area,
                         elevation,cooling_degree_days,heating_degree_days]).reshape(1,-1)
        

        pred = get_prediction(data=data, model=model)

        st.write(f"The RMSE score is:  {pred[0]}")


if __name__ == '__main__':
    main()