# Importing Packages
from cProfile import label
from inspect import stack
from turtle import color
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
    'dragmode': False,
}
config = dict({'displayModeBar': False})

# FIRST HORIZONTAL BAR AT THE HOME PAGE
st.markdown("#### UNDP ACCELERATOR LABS NETWORK")
expander = st.expander('About the Project', expanded=True)

with expander:
    st.write(r"UNDP’s Strategic Plan (2022 – 2025) sets out the ambitious objective to increase access to clean and affordable energy for 500 million people by speeding up investment in distributed renewable energy solutions, especially for those hardest to reach and in crisis context.")
    st.write(r"While large grid and financial flows at scale are essential to reach this goal, our Discover and Deploy Solutions Mapping campaign will explore bottom-up, lead user, frugal and grassroots innovations as a contribution to the sustainable energy access moonshot.")
    st.write(r"Over the course of 4 months, our network of solutions mappers has discovered 359 grassroots energy solutions from across different regions, demographics and energy sources. The discovered solutions help us to signify the importance of already existing grassroots solutions to energy conservation, augmentation, generation, storage, and distribution in early-stage use. Albeit these are often not yet distributed at scale their existence offers valuable insights and trends on how communities are overcoming their own challenges.")
    st.write(r"We need to acknowledge the ingenuity and problem solving capacity found in many communities that in turn can feed into UNDP's programming and contribute to achieving UNDP’s ambitious mission moonshot.")
    


#----------FIRST SECTION----------
st.markdown('#### Where are the solutions coming from?')
st.write('Distribution of Solution per country & per Region')

col_1, col_2 = st.columns(2, gap='small')
with col_1:
    col_1_expander = st.expander("Distribution of Solution per Country", expanded=True)
    with col_1_expander:
        st.write("14% of all Energy solutions came from India alone, making it the country with the highest number of solutions (51). Panama comes next at 8% and Argentina at 7.5% (i.e 28 and 27 solutions respectively).")

    fig_country_count = px.bar(energy_data, 
        x=energy_data['Country'].value_counts(sort=True, ascending = True),
        y=energy_data['Country'].value_counts(sort=True, ascending = True).index,
        title="<b>Distribution of Solutions by Country</b>",
        width = 400,
        height = 500,
        range_x=[0, 51],
        range_y=[28, 48],
        labels=dict(x = 'No. of Solutions', y='Country')
    ).update_layout(default_fig_layout).update_yaxes(tickangle =  -45)
    st.plotly_chart(fig_country_count, use_container_width=False, **{'config': config})   #TODO: Country label not properly displayed


with col_2:
    col_2_expander = st.expander('Distribution of Solutions Per Region', expanded=True)
    with col_2_expander:
        st.write('By regional distribution, the Regional Bureau for Africa (RBA) leads at 33% with a total of 119 solutions. The Regional Bureau for Latin American Countries (RBLAC) and the Regional Bureau for Asia and Pacific (RBAP) closely follow at 29% and 28.7% respectively')
        # st.write('The Regional Bureau for Latin American Countries (RBLAC) and the Regional Bureau for Asia and Pacific (RBAP) closely follow behind')

    fig_region_count = px.bar(
        energy_data,
        x = energy_data['Region'].value_counts(sort=True, ascending = True),
        y=energy_data['Region'].value_counts(sort=True, ascending = True).index,
        title="<b>Distribution of Solutions by Region</b>",
        width = 400,
        height = 450,
        labels=dict(x='No. of Solutions', y='Region'),
    ).update_layout(default_fig_layout)
    st.plotly_chart(fig_region_count, use_container_width=False, **{'config': config})


#------------ SECOND SECTION------------------
st.markdown('---')
st.markdown("#### Prevalence of Energy Sources")
st.write('What type of energy source is more prevalent, what is less? Are there differences per region, and why?')


col_3, col_4 = st.columns(2, gap='medium')
with col_3:
    col_3_expander = st.expander("Distribution of Solutions Across Energy Sources", expanded=True)
    with col_3_expander:
        st.write("The Household Application Category is the most dominant energy source. This category includes solutions that address specific energy needs at household level. eg renewable cooking stoves, energy meters, battery packs, etc.")

    fig_energy_prevalence = px.bar(
        energy_data,
        x=energy_data['Energy Source'].value_counts(sort=True, ascending = True),
        y=energy_data['Energy Source'].value_counts(sort=True, ascending = True).index,
        title="<b>Distribution of Solutions Across<br>Energy Sources</b>",
        width = 400,
        height = 500,
        labels=dict(x='No. of Solutions', y='Energy Source',)
    ).update_layout(default_fig_layout).update_yaxes(tickangle =  -55)
    st.plotly_chart(fig_energy_prevalence, **{'config': config})

with col_4:
    col_4_expander = st.expander("Distribution of Solutions Across Energy Sources and Regions", expanded=True)
    with col_4_expander:
        st.write("Accross all regions, the Household Application energy source still continued to dominate. Coming next is the Solar energy source. On the contrary, the Chemical and Wind sources are the least popular sources recorded.")

    fig_energy_prevalence = px.bar(
        pd.crosstab(index=energy_data['Region'], columns=energy_data['Energy Source'], normalize='index'),
        color_discrete_sequence=px.colors.qualitative.Dark24_r,
        title="<b>Distribution of Solutions Across Energy<br>Sources and Regions</b>",
        width = 450,
        height = 500,
        barmode = 'stack',
        labels=dict(x='Region', y='No. of Energy Solutions')   # NOT UPDATING AXES LABELS YET
    ).update_layout(default_fig_layout)
    st.plotly_chart(fig_energy_prevalence, **{'config': config})



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
        title="<b>Global Spread of Clean Cooking<br>Solutions</b>",
        width = 400,
        height = 400,
        ).update_layout(default_fig_layout, legend=dict(x=0.95))
    st.plotly_chart(fig_glob_com_clean_cooking_regions, **{'config':config})
    
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
            title="<b>Distribution of Clean Cooking Solutions<br>Across Countries</b>"
            # barmode='group',
        ).update_layout(default_fig_layout).update_yaxes(tickangle =  -45)
    fig_glob_com_clean_cooking_countries.update_layout(legend=dict(x=0.5))
    st.plotly_chart(fig_glob_com_clean_cooking_countries, **{'config': config})


# ----------FOURTH SECTION----------
st.markdown('---')
# st.markdown("#### What overall challenges are the solutions <br>addressing or contributing to overcome?", unsafe_allow_html=True)
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
                      width=mask.shape[1]/2, 
                      height=mask.shape[0]/2,
                     mode='RGBA',
                     background_color='rgba(0,0,0,0)',
                     random_state=42,
                     color_func=similar_color)
wordcloud.generate(list_of_tags)
wordcloud = wordcloud.to_file("wordcloud.png")

col_7, col_8, col_9 = st.columns([2, 2, 1,], gap='large')
with col_7:
    st.markdown("#### What overall challenges are the solutions addressing or <br>contributing to overcome?", unsafe_allow_html=True)
    col_7_expander = st.expander("Overall Challenges solutions address", expanded=True)
    with col_7_expander:
        # st.markdown("#### What overall challenges are the solutions addressing or contributing to overcome?")
        st.write(r"Most solutions are set to address global energy needs by offering sustainable, environmental-friendly and renewable energy options.")
    
with col_8:
    col_8_expander = st.expander("Solutions are set to address general energy needs")
    # with col_9_expander:
    st.image('wordcloud.png', output_format='PNG', channels='RGBA')

with col_9:
    st.empty()

# ----------FIFTH SECTION----------
st.markdown('---')
st.markdown("#### Which Sustainable Development Goals are the solutions advancing in particular, and how?")

fig_sdg_solutions = px.bar(sdg_solutions, 
        y=sdg_solutions['SDG Solution'].value_counts(sort=False,),
        x=sdg_solutions['SDG Solution'].value_counts(sort=False,).index,
        title="<b>Distribution of Solutions Across The 17 SDGs</b>",
        width = 450,
        height = 500,
        labels=dict(x='Sustainable Development Goal', y='No. of solutions addressing the SDG')
        # orientation = 'v'
        ).update_layout(default_fig_layout)

col_10, col_11, col_12 = st.columns([2, 2, 1], gap='small')
with col_10:
    col_10_expander = st.expander("SDGs Advanced by Solutions", expanded=True)
    with col_10_expander:
        st.write("Expectedly, SDG 7 (Affordable and Clean Energy) is the most prominent SDG the projects tend to address, followed by SDG 11 (Sustainable Cities and Communities).")
        st.write("Most solutions are proposing a cleaner and more sustainable energy solutions.")

with col_11:
    st.plotly_chart(fig_sdg_solutions, **{'config': config})

with col_12:
    st.empty()


# ----------SIXTH SECTION: Prevalence of Clean Cooking Solutions----------
st.markdown('---')
st.markdown("#### Prevalence of Clean Cooking solutions")

col_13, col_14 = st.columns(2, gap='medium')
with col_13:
    col_13_expander = st.expander("Clean Cooking Solutions", expanded=True)
    with col_13_expander:
        st.write("Clean cooking solutions are less prevalent generally. Just about 23% of all solutions belong to this category. However, the Regional Bureau for Asia and Pacific(RBAP) offer more clean cooking energy solutions than other regions")

# Prevalence of clean cooking solution
    values = energy_data['Clean Cooking'].value_counts(sort=True)
    names = energy_data['Clean Cooking'].value_counts(sort=True).index
    fig = px.pie(energy_data, values = values, names = names,
                title='<b>Prevalence of Clean Cooking Solutions</b>',
                width = 400,
            height = 400,
            # labels=dict(color="Is Clean Cooking Solution")
            ).update_layout(default_fig_layout)

col_15, col_16 = st.columns(2, gap='small')
with col_15:
    st.plotly_chart(fig, **{'config': config})

with col_16:
    # Regional Distribution of Clean Cooking solutions
    is_clean_cooking = energy_data[energy_data['Clean Cooking'].isin([True])]
    clean_cooking_region_dist =  is_clean_cooking.groupby(['Clean Cooking', 'Region']).size().reset_index().pivot(columns='Clean Cooking', index='Region', values=0)
    fig_clean_cooking_region_dist = px.bar(
            x=is_clean_cooking['Region'].value_counts(sort=False,).index,
            y=is_clean_cooking['Region'].value_counts(sort=False,),
            width = 400,
            height = 400,
            title="<b>Distribution of Clean Cooking Solutions <br> Across Regions</b>",
            labels=dict(x='Region', y='No. of Solutions')
            ).update_layout(default_fig_layout)
    st.plotly_chart(fig_clean_cooking_region_dist, **{'config': config})


# ----------SEVENTH SECTION: Maps----------
st.markdown('---')
# st.markdown("#### How can we display the more quantitative information (ratio of IP vs DIY solutions; Prototype vs Product) in an appealing way that signifies the availability of solutions in country? (if applicable)")

col_17, col_18 = st.columns(2, gap="small")
fig_ip_distribution = px.scatter_geo(energy_data,lat='Lat',lon='Long', hover_name="Country", color='Intellectual Property')

fig_ip_distribution.update_geos(showcoastlines = True, showland=True,  landcolor='#FBF8F3',
               showcountries = True, countrycolor='#A49B8C',
               showocean=True, oceancolor='Gray',
               projection = dict(type = 'orthographic'))
fig_ip_distribution.update_layout(title = '<b>Distribution of IP vs DIY Solutions by Country</b>', title_x=0.5,)
fig_ip_distribution.update_layout(legend=dict(y=0.9, x=0))

with col_17:
    st.markdown("#### Ratio of IP vs DIY solutions per Country")
    st.plotly_chart(fig_ip_distribution, **{'config': config})

fig_sol_stage_distribution = px.scatter_geo(energy_data,lat='Lat',lon='Long', 
hover_name="Country", 
title = '<b>Distribution of Solutions by Project Stage</b>',
color='Project Stage',
)

fig_sol_stage_distribution.update_geos(showcoastlines = True, showland=True,  landcolor='#FBF8F3',
                showcountries = True, countrycolor='#A49B8C',
               showocean=True, oceancolor='Gray',
               projection = dict(type = 'orthographic',))
# fig_sol_stage_distribution.update_layout(default_fig_layout)
fig_sol_stage_distribution.update_layout(
    legend=dict(y=0.9, x=0)
)

with col_18:
    st.markdown("#### Ratio of **Prototype** vs **Product** per Country")
    st.plotly_chart(fig_sol_stage_distribution, **{'config': config})
