import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
#------------------------------------------------------
#load saved components of model
model = load_model('artifact/model.keras')
sc = pickle.load(open('artifact/scaler.pkl','rb'))
encoder = pickle.load(open('artifact/encoder.pkl','rb'))
#--------------------------------------------------------

#-------------------------------------------------------

# Add title to the page
st.title("NY :blue[House Prices]")

# Add selection boxes to the page
house_type = st.selectbox("House Type", ('Condo for sale', 'House for sale' ,'Townhouse for sale' ,'Co-op for sale'
 'Multi-family home for sale') )

house_sublocality = st.selectbox("sublocality", ('Manhattan' ,'New York County', 'Richmond County' ,'Kings County', 'New York',
 'East Bronx', 'Brooklyn' ,'The Bronx', 'Queens', 'Staten Island',
 'Queens County', 'Bronx County', 'Coney Island', 'Brooklyn Heights',
 'Jackson Heights' ,'Riverdale', 'Rego Park', 'Fort Hamilton', 'Flushing',
 'Dumbo', 'Snyder Avenue'))

house_bath = st.number_input(label= "No. of Baths", 
                             min_value= 1, 
                             step = 1)

house_beds = st.number_input(label= "No of beds", 
                             min_value= 1, 
                             step = 1)

house_size = st.number_input(label= "size in sq feet", 
                             min_value= 1, 
                             step = 1)
#----------------------------------------------------------------------

### bountry hunter: error in code training
df = pd.DataFrame([[house_type, house_beds, house_bath, house_size, house_sublocality]],
                  columns=['TYPE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'SUBLOCALITY' ])

df_encoded = encoder.transform(df)

df_scaled = sc.transform(df_encoded)

df_predict = model.predict(df_scaled)
df_predict = np.exp(df_predict)[0][0] #[0][0] to remove practs

Error =2.7
# Add Button to the page
if st.button('Calculate'):
    #st.text(f"Accuracy: np.exp(df_predict))
    st.write(f"Price is: :blue[{df_predict}] $ ")
    st.write(f"With Error: :red[{Error}] $")
    