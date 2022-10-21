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
default_fig_layout = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'xaxis': {'showgrid':False,},
    'yaxis':{'showgrid': False,},
    'title_x': 0.5,
    'hovermode': 'x unified',
}

# FIRST HORIZONTAL BAR AT THE HOME PAGE
st.markdown("#### UNDP ACCELERATOR LABS NETWORK")
expander = st.expander('About the Project', expanded=True)

with expander:
    st.write(r"UNDP’s Strategic Plan (2022 – 2025) sets out the ambitious objective to increase access to clean and affordable energy for 500 million people by speeding up investment in distributed renewable energy solutions, especially for those hardest to reach and in crisis context.")
    st.write(r"While large grid and financial flows at scale are essential to reach this goal, our Discover and Deploy Solutions Mapping campaign will explore bottom-up, lead user, frugal and grassroots innovations as a contribution to the sustainable energy access moonshot.")
    st.write("Over the course of 4 months, our network of solutions mappers has discovered 359 grassroots energy solutions from across different regions, demographics and energy sources. The discovered solutions help us to signify the importance of already existing grassroots solutions to energy conservation, augmentation, generation, storage, and distribution in early-stage use. Albeit these are often not yet distributed at scale their existence offers valuable insights and trends on how communities are overcoming their own challenges.")
    st.write("We need to acknowledge the ingenuity and problem solving capacity found in many communities that in turn can feed into UNDP's programming and contribute to achieving UNDP’s ambitious mission moonshot.")
    


#----------FIRST SECTION----------
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
        title="<b>Distribution of Solutions by Country (Top 20)</b>",
        width = 450,
        height = 500,
        range_x=[0, 51],
        range_y=[28, 48]
    ).update_layout(default_fig_layout).update_yaxes(tickangle =  -45)

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
        width = 450,
        height = 400,
    ).update_layout(default_fig_layout)
    st.plotly_chart(fig_region_count, use_container_width=False)


#------------ SECOND SECTION------------------
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
        width = 480,
        height = 500,
    ).update_layout(default_fig_layout).update_yaxes(tickangle =  -55)
    st.plotly_chart(fig_energy_prevalence)

with col_4:
    col_4_expander = st.expander("Distribution of Solutions Across Energy Sources and Regions", expanded=True)
    with col_4_expander:
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry.")

    fig_energy_prevalence = px.bar(
        pd.crosstab(index=energy_data['Region'], columns=energy_data['Energy Source'], normalize='index'),
        color_discrete_sequence=px.colors.qualitative.Dark24_r,
        title="<b>Distribution of Solutions Across Energy Sources and Regions</b>",
        width = 450,
        height = 500,
    ).update_layout(default_fig_layout)
    st.plotly_chart(fig_energy_prevalence)



# ----------THIRD SECTION----------
st.markdown('---')
st.markdown("#### Global Commonalities Across Solutions")
st.write("Typical applications & use cases for solutions")

col_5, col_6 = st.columns([3, 5], gap='small')

with col_5:
    col_5_expander = st.expander('Case Study: Clean Cooking Solutions', expanded=True)
    with col_5_expander:
        st.write("Across all regions, most of the solutions are not deployed for clean cooking")
    stacked_bar_chart = energy_data.groupby(['Clean Cooking', 'Region']).size().reset_index().pivot(columns='Clean Cooking', index='Region', values=0)
    fig_glob_com_clean_cooking_regions = px.bar(
        stacked_bar_chart,
        title="<b>Global Spread of Clean Cooking Solutions</b>",
        width = 400,
        height = 400,
        ).update_layout(default_fig_layout)
    st.plotly_chart(fig_glob_com_clean_cooking_regions)
    
with col_6:
    col_6_expander = st.expander("Case Study: Clean Cooking Solutions", expanded=True)
    with col_6_expander:
        st.write('With the exceptionof India and Rwanda, most solutions are not deployed for clean cooking across all countries studied')
    stacked_bar_chart =  energy_data.groupby(['Clean Cooking', 'Country']).size().reset_index().pivot(columns='Clean Cooking', index='Country', values=0)
    fig_glob_com_clean_cooking_countries = px.bar(
            stacked_bar_chart,
            width = 500,
            height = 900,
            orientation='h',
            # barmode='group',
        ).update_layout(default_fig_layout).update_yaxes(tickangle =  -45)
    st.plotly_chart(fig_glob_com_clean_cooking_countries)


# ----------FOURTH SECTION----------
st.markdown('---')
st.markdown("#### What overall challenges are the solutions addressing or contributing to overcome?")    #TODO: ANSWERS TO THIS SECTION SHALL BE PROVIDED BY NANCY
# Set of Tags across all 5 SDG Solution columns
thematic_tag_list = ['Thematic Tag1', 'Thematic Tag2', 'Thematic Tag3', 'Thematic Tag4', 'Thematic Tag5']
list_of_tags = []
for col in energy_data[thematic_tag_list]:
    get_all_tags = list(energy_data[col])  #All unique values in the column
#     list_of_unique_tags = list(get_unique_tags)
    for tag in get_all_tags:
        tag = str(tag)
        if tag != 'nan':
            #Append the unique entries in each column to the master list 
            list_of_tags.append(tag)  
        
list_of_tags = str(list_of_tags).replace("'", '')


# Set color for the wordcloud
def similar_color(word=None, font_size=None, position=None, orientation=None, font_path=None,  random_state=None):
    h=40
    s=100
    l=random_state.randint(30, 70)
    return "hsl({}, {}%, {}%)".format(h, s, l)

mask = np.array(Image.open(r'mask.png'))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      mask=mask,
                      width=mask.shape[1], 
                      height=mask.shape[0],
                     mode='RGBA',
                     background_color='rgba(0,0,0,0)',
                     random_state=42,
                     color_func=similar_color)
wordcloud.generate(list_of_tags)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='None',)
plt.axis('off')
plt.savefig('wordcloud.png', bbox_inches = 'tight')

col_7, col_8, col_9 = st.columns([1,2,1])
with col_7:
    st.empty()
with col_9:
    st.empty()
with col_8:
    st.image('wordcloud.png', output_format='PNG', channels='RGBA')

# ----------FIFTH SECTION----------
st.markdown('---')
st.markdown("#### Which Sustainable Development Goals are the solutions advancing in particular, and how?")

fig_sdg_solutions = px.bar(sdg_solutions, 
        y=sdg_solutions['SDG Solution'].value_counts(sort=False,),
        x=sdg_solutions['SDG Solution'].value_counts(sort=False,).index,
        title="<b>Distribution of Solutions Across The 17 SDGs</b>",
        width = 450,
        height = 500,
        # orientation = 'v'
        ).update_layout(default_fig_layout)

st.plotly_chart(fig_sdg_solutions)


# ----------SIXTH SECTION: Prevalence of Clean Cooking Solutions----------
st.markdown('---')
st.markdown("#### Looking at the use case of clean cooking solutions, what is their prevalence, distribution, and source of energy?")

# Prevalence of clean cooking solution
values = energy_data['Clean Cooking'].value_counts(sort=True)
names = energy_data['Clean Cooking'].value_counts(sort=True).index
fig = px.pie(energy_data, values = values, names = names,
            title='<b>Prevalence of Clean Cooking Solutions</b>',
            width = 450,
        height = 450,
        ).update_layout(default_fig_layout)

# pie_col1, pie_col2, pie_col3 = st.columns([1,2,1])
# with pie_col2:
st.plotly_chart(fig)

# Regional Distribution of Clean Cooking solutions
is_clean_cooking = energy_data[energy_data['Clean Cooking'].isin([True])]
clean_cooking_region_dist =  is_clean_cooking.groupby(['Clean Cooking', 'Region']).size().reset_index().pivot(columns='Clean Cooking', index='Region', values=0)
fig_clean_cooking_region_dist = px.bar(
        x=is_clean_cooking['Region'].value_counts(sort=False,).index,
        y=is_clean_cooking['Region'].value_counts(sort=False,),
        # width = 500,
        # height = 900
        ).update_layout(default_fig_layout)
st.plotly_chart(fig_clean_cooking_region_dist)


# ----------SEVENTH SECTION: Maps----------
st.markdown('---')
st.markdown("#### How can we display the more quantitative information (ratio of IP vs DIY solutions; Prototype vs Product) in an appealing way that signifies the availability of solutions in country? (if applicable)")
fig_ip_distribution = px.scatter_geo(energy_data,lat='Lat',lon='Long', hover_name="Country", color='Intellectual Property')

fig_ip_distribution.update_geos(showcoastlines = True, showland=True,  landcolor='#FBF8F3',
                showcountries = True, countrycolor='#A49B8C',
               showocean=True, oceancolor='Gray',
               projection = dict(type = 'orthographic'))
fig_ip_distribution.update_layout(title = 'Distribution of IP vs DIY Solutions by Country', title_x=0.45,)
st.plotly_chart(fig_ip_distribution)

fig_sol_stage_distribution = px.scatter_geo(energy_data,lat='Lat',lon='Long', hover_name="Country", color='Intellectual Property')

fig_sol_stage_distribution.update_geos(showcoastlines = True, showland=True,  landcolor='#FBF8F3',
                showcountries = True, countrycolor='#A49B8C',
               showocean=True, oceancolor='Gray',
               projection = dict(type = 'orthographic',))
fig_sol_stage_distribution.update_layout(title = 'Distribution of IP vs DIY Solutions by Country', title_x=0.45,)