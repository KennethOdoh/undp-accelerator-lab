# Importing Packages
from inspect import stack
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# Control Pandas behaviours
pd.options.mode.chained_assignment = None

# import translators as ts


# Page configurations
st.set_page_config(
    page_title = 'UNDP Accelerator Labs Network',
    page_icon = "undp-logo-blue.png",
    layout = 'wide',
)

# load data set
energy_data = pd.read_csv("energy_data_cleaned.csv")
sdg_solutions = pd.read_csv('sdg-solutions.csv')
sdg_meaning = pd.read_excel('sdg_goals.xlsx', usecols = ['sdg_goal_description'])

# Change data types of listed columns
energy_data_dtypes = {
    'Date': 'datetime64[ns]',
    'Energy Source': 'category',
    'Clean Cooking': 'category',
    'Can Train Others' : 'bool',
    'Market Ready' : 'bool',
    'Requires Advanced Order' : 'bool',
    'Lat' : 'float64',
    'Long' : 'float64',
    'Project Stage' : 'category',
    'Intellectual Property' :'category',
    'SDG 1': 'bool',
    'SDG 2': 'bool',
    'SDG 3' : 'bool',
    'SDG 4' : 'bool',
    'SDG 5' : 'bool',
    'SDG 6' : 'bool',
    'SDG 7' : 'bool',
    'SDG 8' : 'bool',
    'SDG 9' : 'bool',
    'SDG 10' : 'bool',
    'SDG 11' : 'bool',
    'SDG 12' : 'bool',
    'SDG 13' : 'bool',
    'SDG 14' : 'bool',
    'SDG 15' : 'bool',
    'SDG 16' : 'bool',
    'SDG 17' : 'bool',
    'Country' : 'object',
    'Region' : 'object',
}

sdg_solutions_dtype = {
    'SDG Solution': 'category',
    'Goal Description': 'category',
}

energy_data = energy_data.astype(energy_data_dtypes)
sdg_solutions = sdg_solutions.astype(sdg_solutions_dtype)


# Define global variables Here
base_color = "#418FDE"

# FIRST HORIZONTAL BAR AT THE HOME PAGE
st.markdown("#### UNDP ACCELERATOR LABS NETWORK")
expander = st.expander('About the Project')
st.markdown('---')

with expander:
    st.write(r"UNDP’s Strategic Plan (2022 – 2025) sets out the ambitious objective to increase access to clean and affordable energy for 500 million people by speeding up investment in distributed renewable energy solutions, especially for those hardest to reach and in crisis context.")
    st.write(r"While large grid and financial flows at scale are essential to reach this goal, our Discover and Deploy Solutions Mapping campaign will explore bottom-up, lead user, frugal and grassroots innovations as a contribution to the sustainable energy access moonshot.")
    st.write("Over the course of 4 months, our network of solutions mappers has discovered 359 grassroots energy solutions from across different regions, demographics and energy sources. The discovered solutions help us to signify the importance of already existing grassroots solutions to energy conservation, augmentation, generation, storage, and distribution in early-stage use. Albeit these are often not yet distributed at scale their existence offers valuable insights and trends on how communities are overcoming their own challenges.")
    st.write("We need to acknowledge the ingenuity and problem solving capacity found in many communities that in turn can feed into UNDP's programming and contribute to achieving UNDP’s ambitious mission moonshot.")
    


st.markdown('#### Where are the solutions coming from?')
st.write('Distribution of Solution per country & per Region')

col_1, col_2 = st.columns(2, gap='small')
with col_1:
    col_1_expander = st.expander("Distribution of Solution per Country", expanded=True)
    with col_1_expander:
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry.")

    fig_country_count = px.bar(energy_data, 
        x=energy_data['Country'].value_counts(sort=True, ascending = True),
        y=energy_data['Country'].value_counts(sort=True, ascending = True).index,
        title="<b>Distribution of Solutions by Country</b>",
        width = 700,
        height = 500,
    )

    st.plotly_chart(fig_country_count, use_container_width=False)   #TODO: Country label not properly displayed

with col_2:
    col_2_expander = st.expander('Distribution of Solutions Per Region', expanded=True)
    with col_2_expander:
        st.write('Lorem Ipsum is simply dummy text of the printing and typesetting industry.')

    fig_region_count = px.bar(
        energy_data,
        x = energy_data['Region'].value_counts(sort=True, ascending = True),
        y=energy_data['Region'].value_counts(sort=True, ascending = True).index,
        title="<b>Distribution of Solutions by Region</b>",
        width = 500,
        height = 350,
    )
    st.plotly_chart(fig_region_count, use_container_width=False)

st.markdown('---')
st.markdown("#### Prevalence of Energy Sources")
st.write('What type of energy source is more prevalent, what is less? Are there differences per region, and why?')


col_3, col_4 = st.columns(2, gap='medium')
with col_3:
    col_3_expander = st.expander("Distribution of Solutions Across Energy Sources", expanded=True)
    with col_3_expander:
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry.")

    fig_energy_prevalence = px.bar(
        energy_data,
        x=energy_data['Energy Source'].value_counts(sort=True, ascending = True),
        y=energy_data['Energy Source'].value_counts(sort=True, ascending = True).index,
        title="<b>Distribution of Solutions Across Energy Sources</b>",
        width = 700,
        height = 500,
    )
    st.plotly_chart(fig_energy_prevalence)

with col_4:
    col_4_expander = st.expander("Distribution of Solutions Across Energy Sources and Regions", expanded=True)
    with col_4_expander:
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry.")

    fig_energy_prevalence = px.bar(
        pd.crosstab(index=energy_data['Region'], columns=energy_data['Energy Source'], normalize='index'),
        color_discrete_sequence=px.colors.qualitative.Dark24_r,
        title="<b>Distribution of Solutions Across Energy Sources and Regions</b>",
        width = 500,
        height = 500,
    )
    st.plotly_chart(fig_energy_prevalence)

# st.dataframe(energy_data)
# st.dataframe(sdg_solutions)
# st.dataframe(sdg_meaning)

# st.write(Counter(energy_data['Country']))