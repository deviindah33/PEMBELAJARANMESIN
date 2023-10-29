import pickle
import numpy as np
import streamlit as st

model = pickle .load(open('HIV.sav','rb'))

st.title('Prediksi Terkena Virus HIV')

col1, col2 = st.columns(2)

with col1:
   Age	= st.number_input('Umur Pasien')
   Marital_Staus = st.number_input('Status Pernikahan')
   STD	= st.number_input('Penyakit Menular Sexual')
   Educational_Background	= st.number_input('Latar Belakang Pendidikan')
   HIV_TEST_IN_PAST_YEAR	= st.number_input('Test HIV Tahun Lalu')
with col2:
   AIDS_education	= st.number_input('Pendidikan AIDS')
   Places_of_seeking_sex_partners	= st.number_input('Tempat Mencari Pasangan Seks')
   SEXUAL_ORIENTATION	= st.number_input('Orientasi Sexual')
   Drug_taking = st.number_input('Minum Obat')

predik = ''
if st.button('Hasil Prediksi'):
    predik = model.predict([[Age, Marital_Staus, STD, Educational_Background, HIV_TEST_IN_PAST_YEAR, 
    AIDS_education, Places_of_seeking_sex_partners, SEXUAL_ORIENTATION, Drug_taking
]])

    if(predik[0] == 1):
        predik = 'Kemungkinan Pasien tidak terkena Virus HIV'
    else:
        predik = 'Kemungkinan Pasien terkena Virus HIV'
st.success(predik)
