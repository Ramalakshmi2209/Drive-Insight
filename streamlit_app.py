import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def streamlit_app():
    st.title("Driver Hours Prediction")

    # User input for driver_id and date
    driver_id = st.number_input("Enter Driver ID", value=1, step=1)
    date = st.date_input("Select Date", value=datetime.today())

  
    if st.button("Predict"):
        try:
            
            response = requests.post("http://localhost:5000/", data={"driver_id": driver_id, "date": str(date)})

            
            soup = BeautifulSoup(response.text, 'html.parser')

            
            predicted_hours = soup.body.find('h1').next_sibling.strip()
            demand_status = soup.body.find_all('br')[-1].next_sibling.strip()

            
            st.markdown(f"Predicted Online Hours: {predicted_hours}")
            st.markdown(f"Demand Status: {demand_status}")

        except Exception as e:
            st.error(f"Error: {str(e)}")


streamlit_app()
