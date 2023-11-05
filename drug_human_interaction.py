import streamlit as st
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

st.title("Patient Information Form")

st.warning("Disclaimer: This is a preliminary analysis. Consult a physician for more information.")

drug_consumption = st.text_input("Drug Being Consumed", "")

medical_history = st.text_input("Medical History", "")

pregnant = st.checkbox("Pregnant")
alcohol_consumption = st.checkbox("Alcohol Consumption")
breastfeeding = st.checkbox("Breastfeeding")

if st.button("Submit"):
    base_url = "https://api.fda.gov/drug/label.json"
#drug_name = "Ibuprofen"
    query_params = {
        "search": f"openfda.generic_name:{drug_consumption}",
        "limit": 10 
    }
    response = requests.get(base_url, params=query_params)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"API request failed with status code: {response.status_code}")
    drug_data=data['results']
    drug_text=drug_data[0].get("warnings")[0]
    def preprocess_text(text):
        words = text.split()
        punctuations = r'[,.":/[\]()\']'
        word_list = [re.sub(punctuations, '', word).lower() for word in words]
        filtered_words = [word for word in word_list if word not in stopwords.words('english')]
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        unique_words = list(set(stemmed_words))
        return unique_words
    drug_effects=preprocess_text(drug_text)
    Medhist=preprocess_text(medical_history)
    setlist1=set(drug_effects)
    setlist2=set(Medhist)
    matching_list=list(setlist1.intersection(setlist2))
    w=""
    if pregnant:
        val=["pregnant","pregnanc","unborn"]
        count=0
        for value in val:
            if value in setlist1:
                w=value
                count=count+1
        if count>0:
            st.warning("Patient is pregnant and should not take this drug.")

    if alcohol_consumption:
        val="alcohol"
        if val in setlist1:
            st.warning("Patient consumes alcohol and should not take this drug.")

    if breastfeeding:
        val="breast-feed"
        if val in setlist1:
            st.warning("Patient is breastfeeding and should not take this drug.")

    matching_count=len(setlist1.intersection(setlist2))

    if len(matching_list) > 0:
        st.warning("The existing medical condition and the drug could cause effects. Please consult a physician for further inquiry.")

    else:
        st.success('Its safe to take the medicine given your medical history.')