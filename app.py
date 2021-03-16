import os
import dash
import pickle
import numpy as np
import pandas as pd
from datetime import date
import plotly.express as px
from dash import callback_context
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import requests
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from keras.models import model_from_json

os.chdir(os.getcwd())

response = requests.get('https://github.com/basvanderbijl/OBPAmsterdamTrafficForecast/raw/main/data/datav12.csv')
df_dash = pd.read_csv(response.content)

locations = sorted(df_dash['naam_meetlocatie_mst'].unique(), key=len)
locations.reverse()

location_mapping = df_dash[['id_meetlocatie', 'naam_meetlocatie_mst']]
location_mapping = location_mapping.drop_duplicates()

# with open('C:/Users/Yorran Reurich/OneDrive/VU/Master/POBP/holt-winters-test.pickle', 'rb') as f:
#     HW_model = pickle.load(f)
with open('pickles/tbats_hourly_all.pickle', 'rb') as file:
    tbats_models_hourly = pickle.load(file)
with open('pickles/tbats_daily_all.pickle', 'rb') as file:
    tbats_models_daily = pickle.load(file)
# with open('C:/Users/stanv/OneDrive/Documenten/Master Business Analytics/OBP project/LSTM_daily_models_test.pickle', 'rb') as file:
#     LSTM_models = pickle.load(file)

earliest_date = date(2016,1,1)
latest_date = date(2022,12,31)

algorithm_list = ['TBATS hourly', 'TBATS daily', 'Seasonal Naive']
area_locations = df_dash.groupby('naam_meetlocatie_mst').agg({'start_locatie_latitude': 'last', 'start_locatie_longitude': 'last'}).reset_index()

img_logo_vu = 'Vu_logo_blauw.png'
#https://www.vu.nl/nl/Images/VUlogo_NL_Blauw_HR_RGB_tcm289-201375.png
img_logo_ams = 'Gemeente-Amsterdam-scaled.png'
#https://happyoffice.nl/wp-content/uploads/2015/12/Gemeente-Amsterdam-scaled.png

loading_sign = 'dot'
link_to_user_manual = 'https://docs.google.com/document/d/e/2PACX-1vTZUVcRZwqeFDgV--WNWVhbTztOqqeVwlik_8gYfhv2xYCrbdtBa9g7LhkyYEqN4-cf-FwiKYLpKs0d/pub'

colors = {
'background_full_dashboard': '#f8f8ff', #fffaf0  #Gebroken wit
'background_white': '##f8f8ff',
'background_objects': '#F1F1F1',#'#addde6',
'text_color_title': '#FFD700', #Yellow/gold-ish
'font_color_black': '#111111',
'text_color_graphs': '#778899', #Gray
'Tabs_background': '#e6e6e6',
'color_VU': '#0089cf',
'color_Amst': '#da121a'}

tab_style = {
'backgroundColor':  colors['background_objects'],
'borderWidth': '0px',
'borderRadius': '20px',
'fontSize': '20px',
'padding': '15px'}

tab_selected_style = {
'backgroundColor': '#0089cf',
'color': '#FFFFFF',
'borderWidth': '0px',
'borderRadius': '20px', 
'fontSize': '20px',
'padding': '15px',
'fontWeight': 'bold'}

tab_1_box_style = {
'width': '50%',
'padding': '20px',
'backgroundColor': colors['background_objects'],
'borderRadius': '5px', 
'overflow': 'hidden',
'box-shadow': '2px 2px 2px lightgrey'}

intro_style = {
'width': '75%',
'textAlign': 'center',
'color': colors['text_color_graphs'],
'font-size': '18px',
'marginBottom': '80px'}

filter_style = {
'width': '32%',
'padding': '20px',
'backgroundColor': colors['background_objects'],
'borderRadius': '5px',
'box-shadow': '2px 2px 2px 2px lightgrey'}

add_white_space = html.Div(className='row3', style={'margin-bottom': '40px'})
tab_end_bar = html.Div(style={'display': 'flex'}, children=[
    html.Div(style={'backgroundColor': colors['color_Amst'], 'height': 60, 'width': '1%'}),
    html.Div(style={'backgroundColor': colors['color_Amst'], 'height': 60, 'width': '10%', 'backgroundColor': colors['color_Amst']}, children=[
        html.Div(style={'margin-bottom': '4px'}),
        html.Img(src='https://happyoffice.nl/wp-content/uploads/2015/12/Gemeente-Amsterdam-scaled.png',style={'height': 55,'backgroundColor': colors['color_Amst']})]),
    html.Div(style={'backgroundColor': colors['color_Amst'], 'height': 60, 'width': '38.75%'}),
    html.Div(style={'backgroundColor': colors['background_full_dashboard'], 'height': 60, 'width': '0.5%'}),
    html.Div(style={'backgroundColor': colors['color_VU'], 'height': 60, 'width': '38.75%'}),  
    html.Img(src='https://www.vu.nl/nl/Images/VUlogo_NL_Blauw_HR_RGB_tcm289-201375.png',style={'height': 60, 'width': '10%'}),
    html.Div(style={'backgroundColor': colors['color_VU'], 'height': 60, 'width': '1%'})])

code_tab_0 = dcc.Tab(label='Home page', value='tab_0', style=tab_style, selected_style=tab_selected_style, children=[
    html.Div(style={'marginTop': '40px'}),
    html.Center(
        html.Div(
            children=['This dashboard can be used as a Decision Support System (DSS). An extended user manual can be found ',
            html.A('here', href=link_to_user_manual, target="_blank"), '. ',
            '''This DSS is developed under the supervision of Thomas Koch at the Vrije Universiteit van Amsterdam. 
            The goal of this dashboard is to give insights into existing and future traffic intensities in and around the Amsterdam area. 
            It is possible to switch between the tabs above to see different formats wherein the data is shown. Filters can be adjusted in each tab to help the user with their analysis. 
            An explanation of each tab is given in the boxes below. 
            Tips and bugs can be communicated via this e-mail address: stan.van.loon@vu.nl'''],
            style=intro_style)),
    html.Div(className='row', children=[
        html.Div(style={'margin-left': '20px'}),
        html.Div([
            html.H4('Tab 1: Map', style={'margin-top': '0px'}),
            dcc.Markdown("""
                In this first tab it is possible to see the traffic intensity for each specific area individually. 
                The first filter in the top left-hand corner, an area can be selected and will be shown in the right graph. 
                The filter in the middle is for choosing a specific time frame on a daily basis. It is also possible to change the time frame when clicking on the graph and on the rangeslider. 
                The last filter in the top right-hand corner, here a specific prediction method can be selected. This prediction method will be used to predict traffic intensity for the selected area. 
                To see a prediction further into the future, the second filter can be used.""")],
            style=tab_1_box_style,
            className='twelve columns pretty_container'),
        html.Div(style={'margin-left': '15px'}),
        html.Div([
            html.H4('Tab 2: Area Comparison', style={'margin-top': '0px'}),
            dcc.Markdown("""
                In this second tab it is possible to compare traffic intensity for different areas. 
                The first filter in the top left-hand corner, multiple areas can be selected and will be shown in the graph below. 
                The filter in the middle is for choosing a specific time frame on a daily basis. It is also possible to change the time frame when clicking on the graph and on the rangeslider. 
                The last filter in the top right-hand corner, here a specific prediction method can be selected. This prediction method will be used to predict traffic intensity for the selected area. 
                To see a prediction further into the future, the second filter can be used.""")],
            style=tab_1_box_style, 
            className='twelve columns pretty_container'),
        html.Div(style={'margin-left': '15px'}),
        html.Div([
            html.H4('Tab 3: Algorithm Comparison', style={'margin-top': '0px'}),
            dcc.Markdown("""
                In this third tab it is possible to compare prediction of traffic intensity of multiple prediction methods for a specific area. 
                The first filter in the top left-hand corner, an area can be selected and will be shown in the graph below. 
                The filter in the middle is for choosing a specific time frame on a daily basis. It is also possible to change the time frame when clicking on the graph and on the rangeslider. 
                The last filter in the top right-hand corner, here multiple prediction methods can be selected. This prediction method will be used to predict traffic intensity for the selected area. 
                To see a prediction further into the future, the second filter can be used.""")],
            style=tab_1_box_style,
            className='twelve columns pretty_container'),
        html.Div(style={'margin-left': '20px'})],
        style={'display': 'flex'}),
    add_white_space,
    tab_end_bar
    ])

code_tab_1 = dcc.Tab(label='Map', value='tab_1', style=tab_style, selected_style=tab_selected_style, children=[
    html.Div(children=[
        html.Div(className='row', children=[
            html.Div([
                html.Center([html.Div('Change location', style={'font-size': '20px'})]),
                html.Div(style={'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='area',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in locations],
                    value=locations[0], #'RWS01_MONIBAS_0011hrr0085ra', #0011hrr0085ra
                    style={'textAlign': 'center'})],
                style=filter_style),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.Center([html.Div('Change time window', style={'font-size': '20px'})]),
                html.Div(style={'marginBottom': '20px'}),
                html.Center(dcc.DatePickerRange(
                    id='dates',
                    start_date=earliest_date,
                    end_date=latest_date,
                    display_format='D/M/Y',
                    style={'background-color': colors['background_objects']}))],
                style=filter_style),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.Center(html.Div('''Select prediction method''', style={'font-size': '20px'})),
                html.Div(style={'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='algorithms',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in algorithm_list],
                    value=algorithm_list[0],
                    style={'textAlign': 'center'})],
                style=filter_style)],
            style={'display': 'flex', 'marginBottom': '30px', 'marginRight': '20px', 'marginLeft': '20px'}),

        html.Div(style={'marginBottom': '30px'}),
        html.Div(className='row', children=[
            html.Div(children=[
                html.H6('Average Speed'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='avg_speed1', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.H6('Average Intensity'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='avg_intensity1', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.H6('Mean Absolute Error'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='mae1', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px', 
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey',
                'marginRight': '20px'})],
            style={'display': 'flex', 'marginBottom': '30px', 'marginRight': '20px', 'marginLeft': '20px'}),
        html.Div(style={'marginBottom': '30px'}),

        html.Div(className='row3', children=[
            html.Div(className='two columns', children=[dcc.Graph(id='map')], style={
                'width': '46.65%',
                'marginLeft': '20px',
                'padding': '20px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(className='two columns', children=[
                dcc.Loading(type=loading_sign, children=[
                    dcc.Graph(id='graph')
                    ])
                ], style={
                'width': '46.65%',
                'marginRight': '20px',
                'padding': '20px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'})
            ], style={'width': '100%', 'display': 'inline-block'}),
        add_white_space,
        tab_end_bar
        ])
    ])

code_tab_2 = dcc.Tab(label='Location comparison', value='tab_2', style=tab_style, selected_style=tab_selected_style, children=[
    html.Div([
        html.Div(className='row', children=[
            html.Div([
                html.Center([html.Div('''Change location''', style={'font-size': '20px'})]),
                html.Div(style={'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='area2',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in locations],
                    value=[locations[0], locations[1]],
                    style={'textAlign': 'center'},
                    multi=True)],
                style=filter_style),
            html.Div(style={'margin-left': '15px'}),
            html.Div(children=[
                html.Center([html.Div('''Change time window''', style={'font-size': '20px'})]),
                html.Div(style={'marginBottom': '20px'}),
                html.Center(dcc.DatePickerRange(
                    id='dates2',
                    start_date=earliest_date,
                    end_date=latest_date,
                    display_format='D/M/Y',
                    start_date_placeholder_text='M/D/YYYY',
                    style={'display': 'inline-block'}))],
                style=filter_style),
            html.Div(style={'margin-left': '15px'}),
            html.Div(children=[
                html.Center(html.Div('''Select prediction method''', style={'font-size': '20px'})),
                html.Div(style={'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='algorithms2',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in algorithm_list],
                    value=algorithm_list[0],
                    style={'textAlign': 'center'})],
                style=filter_style)],
            style={'display': 'flex', 'marginBottom': '30px', 'marginRight': '20px', 'marginLeft': '20px'}),

        html.Div(style={'marginBottom': '30px'}),
        html.Div(className='row2', children=[
            html.Div(children=[
                html.H6('Average Speed'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='avg_speed2', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.H6('Average Intensity'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='avg_intensity2', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.H6('Mean Absolute Error'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='mae2', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px', 
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey',
                'marginRight': '20px'})],
            style={'display': 'flex', 'marginBottom': '30px', 'marginRight': '20px', 'marginLeft': '20px'}),
        html.Div(style={'marginBottom': '30px'}),

        dcc.Loading([dcc.Graph(id='graph2',
            style={
            'padding': '20px',
            'backgroundColor': colors['background_objects'],
            'borderRadius': '5px',
            'overflow': 'hidden',
            'marginLeft': '20px',
            'box-shadow': '2px 2px 2px 2px lightgrey',
            'marginRight': '20px'})],
        type=loading_sign)]),
    add_white_space,
    tab_end_bar
    ])

code_tab_3 = dcc.Tab(label='Algorithm comparison', value='tab_3', style=tab_style, selected_style=tab_selected_style, children=[
    html.Div([
        html.Div(className='row', children=[
            html.Div([
                html.Center([html.Div('Change location', style={'font-size': '20px'})]),
                html.Div(style={'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='area3',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in locations],
                    value=locations[0], #'RWS01_MONIBAS_0011hrr0085ra', # #0011hrr0085ra
                    style={'textAlign': 'center'})
                ], style=filter_style),
            html.Div(style={'margin-left': '15px'}),
            html.Div(children=[
                html.Center([html.Div('Change time window', style={'font-size': '20px'})]),
                html.Div(style={'marginBottom': '20px'}),
                html.Center(dcc.DatePickerRange(
                    id='dates3',
                    start_date=earliest_date,
                    end_date=latest_date,
                    display_format='D/M/Y',
                    start_date_placeholder_text='M/D/YYYY',
                    style={'display': 'inline-block', 'width': '100%'}))],
                style=filter_style),
            html.Div(style={'margin-left': '15px'}),
            html.Div(children=[
                html.Center(html.Div('''Select prediction method''', style={'font-size': '20px'})),
                html.Div(style={'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='algorithms3',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in algorithm_list],
                    value=[algorithm_list[0], algorithm_list[1]],
                    style={'textAlign': 'center'},
                    multi=True)],
                style=filter_style)],
            style={'display': 'flex', 'marginBottom': '30px', 'marginRight': '20px', 'marginLeft': '20px'}),

        html.Div(style={'marginBottom': '30px'}),
        html.Div(className='row2', children=[
            html.Div(children=[
                html.H6('Average Speed'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='avg_speed3', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.H6('Average Intensity'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='avg_intensity3', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px',
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey'}),
            html.Div(style={'marginLeft': '15px'}),
            html.Div(children=[
                html.H6('Mean Absolute Error'),
                dcc.Loading(type=loading_sign, children=[html.Center([html.Div(dcc.Graph(id='mae3', style={'height': '50px', 'marginBottom': '5px', 'fontWeight': 'bold'}))])])],
                style={
                'width': '32%',
                'padding': '10px', 
                'backgroundColor': colors['background_objects'],
                'borderRadius': '5px',
                'box-shadow': '2px 2px 2px 2px lightgrey',
                'marginRight': '20px'})],
            style={'display': 'flex', 'marginBottom': '30px', 'marginRight': '20px', 'marginLeft': '20px'}),
        html.Div(style={'marginBottom': '30px'}),

        dcc.Loading([dcc.Graph(id='graph3',
            style={
            'marginLeft': '20px',
            'marginRight': '20px',
            'borderRadius': '5px',
            'overflow': 'hidden',
            'box-shadow': '2px 2px 2px 2px lightgrey'})],
        type=loading_sign)]),
    add_white_space,
    tab_end_bar
    ])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Center(
    html.Div(id='dashboard', style={'width': '99%', 'height': '100%', 'backgroundColor': colors['background_full_dashboard']}, children=[
        html.Div(className='row', children=[
            html.Center([html.Label('Time series forecasting with Amsterdam traffic data', style={'fontSize': '48px', 'margingTop': '20px', 'marginBottom': '20px'})]),
            html.Center(dcc.Tabs(id='tabs-styled-with-inline', value='tab_0', children=[
                code_tab_0,
                code_tab_1,
                code_tab_2,
                code_tab_3],
                style={'height': '60px', 'width': '97.5%', 'marginBottom': '30px', 'marginLeft': '20px', 'marginRight': '20px', 'backgroundColor': colors['background_objects'], 'borderRadius': '20px'}))],
            style={'width': '100%', 'height': '100%'}),
        html.Div(id='tabs-content-inline')]))


def make_kpi(value):
    return {
    "data": [{
    "type": "indicator",
    "value": value,
    }], "layout": {
    "template": {"layout": {"paper_bgcolor": 'rgba(0,0,0,0)', "plot_bgcolor": 'rgba(0,0,0,0)'}},
    "height": 60,
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0}}}

def tbats_predict_hourly(df, area_name_prediction, end_date):
    model = tbats_models_hourly[area_name_prediction]
    tbats_fitted = pd.DataFrame(np.exp(model.y_hat), columns=['predicted_value'])
    predicted_days = (end_date - max(df['start_meetperiode'])).days
    tbats_fitted['actual'] = df['intensiteit_anyVehicle']

    tbats_fitted = tbats_fitted.dropna()
    mae = mean_absolute_error(tbats_fitted['predicted_value'], tbats_fitted['actual'])
    
    tbats_predicted = pd.DataFrame(np.exp(model.forecast(steps=predicted_days*24)), columns=['predicted_value'])
    tbats_results = pd.concat([tbats_fitted, tbats_predicted], axis=0).reset_index(drop=True)

    df_empty_calendar = pd.DataFrame(pd.date_range(start=min(df['start_meetperiode']),
        end=end_date, freq='H'), columns=['start_meetperiode'])

    df = pd.merge(df_empty_calendar, df, how='left', on=['start_meetperiode'])
    df['prediction'] = tbats_results['predicted_value']

    return df, mae

def tbats_predict_daily(df, area_name_prediction, end_date):
    df = df.set_index('start_meetperiode').groupby([pd.Grouper(freq='D')])['intensiteit_anyVehicle'].sum().reset_index()

    model = tbats_models_daily[area_name_prediction]
    tbats_fitted = pd.DataFrame(model.y_hat, columns=['predicted_value'])
    predicted_days = (end_date - max(df['start_meetperiode'])).days

    mae = mean_absolute_error(tbats_fitted['predicted_value'], df['intensiteit_anyVehicle'])
    
    tbats_predicted = pd.DataFrame(model.forecast(steps=predicted_days), columns=['predicted_value'])
    tbats_results = pd.concat([tbats_fitted, tbats_predicted], axis=0).reset_index(drop=True)

    df_empty_calendar = pd.DataFrame(pd.date_range(start=min(df['start_meetperiode']),
        end=end_date, freq='D'), columns=['start_meetperiode'])

    df = pd.merge(df_empty_calendar, df, how='left', on=['start_meetperiode'])
    df['prediction'] = tbats_results['predicted_value']

    return df, mae

@app.callback(
    Output('map', 'figure'),
    Output('graph', 'figure'),
    Output('avg_speed1', 'figure'),
    Output('avg_intensity1', 'figure'),
    Output('mae1', 'figure'),
    Input('area', 'value'),
    Input('dates', 'start_date'),
    Input('dates', 'end_date'),
    Input('map', 'clickData'),
    Input('algorithms', 'value'))
def update_tab_1(area_dropdown, start_date, end_date, clickData, algorithm):
    trigger = callback_context.triggered[0]['prop_id']
    if trigger == 'map.clickData':
        area = clickData['points'][0]['hovertext']
    else:
        area = area_dropdown

    #area = 'RWS01_MONIBAS_0011hrr0085ra'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_dash_location = df_dash.loc[(df_dash['naam_meetlocatie_mst'] == area) & (pd.to_datetime(df_dash['start_meetperiode']) >= start_date) & (pd.to_datetime(df_dash['start_meetperiode']) <= end_date)]
    df_dash_location['start_meetperiode'] = pd.to_datetime(df_dash_location['start_meetperiode'])
    df_dash_location = df_dash_location[['start_meetperiode', 'intensiteit_anyVehicle']].set_index('start_meetperiode').asfreq('H').reset_index()

    area_name_prediction = location_mapping.loc[location_mapping['naam_meetlocatie_mst'] == area, 'id_meetlocatie'].item()
    if algorithm == 'TBATS hourly':
        total_df, mae = tbats_predict_hourly(df_dash_location, area_name_prediction, end_date)
    if algorithm == 'TBATS daily':
        total_df, mae = tbats_predict_daily(df_dash_location, area_name_prediction, end_date)

    avg_speed1 = make_kpi(np.mean(df_dash_location['intensiteit_anyVehicle'])/10)
    avg_intensity1 = make_kpi(np.mean(df_dash_location['intensiteit_anyVehicle']))
    mae1 = make_kpi(round(mae,2))

    area_locations['color'] = np.where(area_locations['naam_meetlocatie_mst'] != area, colors['color_VU'], colors['color_Amst'])
    fig = px.scatter_mapbox(
        area_locations,
        lat='start_locatie_latitude',
        lon='start_locatie_longitude',
        hover_name='naam_meetlocatie_mst',
        hover_data='',
        color='color',
        zoom=9.5)
    fig.update_layout(mapbox_style='open-street-map', showlegend=False)
    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b' :0})

    data_tab_1=[]
    data_tab_1.append(go.Scatter(
         x=total_df.start_meetperiode.tolist(), 
         y=total_df.prediction.tolist(),
         name='Predicted',
         line=dict(color=colors['color_Amst'])))

    data_tab_1.append(go.Scatter(
        x=total_df.start_meetperiode.tolist(),
        y=total_df.intensiteit_anyVehicle.tolist(),
        name='Actual',
        line=dict(dash='dot', color = colors['color_VU'])))

    layout = go.Layout(
        title=f'<b>{area}</b>',
        xaxis={'title': 'Time'},
        yaxis={'title': 'Intensity'},
        margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
        hovermode='closest')

    fig2 = go.Figure(data=data_tab_1, layout=layout)
    fig2.update(layout_xaxis_rangeslider_visible=True)
    fig2.update_layout(
        hovermode='closest', xaxis_title='Time',
        yaxis_title='Traffic Intensity',
        title_font_color='black',
        title_font_size=24,
        paper_bgcolor=colors['background_objects'],
        plot_bgcolor=colors['background_objects'])

    return fig, fig2, avg_speed1, avg_intensity1, mae1

@app.callback(
    Output('graph2', 'figure'),
    Output('avg_speed2', 'figure'),
    Output('avg_intensity2', 'figure'),
    Output('mae2', 'figure'),
    Input('area2', 'value'),
    Input('dates2', 'start_date'),
    Input('dates2', 'end_date'),
    Input('algorithms2', 'value'))
def update_tab_2(areas, start_date, end_date, algorithm):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data_tab_2 = []
    df_dash_filter = df_dash.loc[(pd.to_datetime(df_dash['start_meetperiode']) >= start_date) & (pd.to_datetime(df_dash['start_meetperiode']) <= end_date)]
    df_dash_filter = df_dash_filter.loc[df_dash_filter['naam_meetlocatie_mst'].isin([area for area in areas])]
    groups = df_dash_filter.groupby(by='naam_meetlocatie_mst')

    avg_speed2 = make_kpi(np.mean(df_dash_filter['intensiteit_anyVehicle'])/10)
    avg_intensity2 = make_kpi(np.mean(df_dash_filter['intensiteit_anyVehicle']))
    mae2 = make_kpi(np.mean(df_dash_filter['intensiteit_anyVehicle'])/2)

    for group, dataframe in groups:
        trace = go.Scatter(
            x=dataframe.start_meetperiode.tolist(), 
            y=dataframe.intensiteit_anyVehicle.tolist(),
            opacity=0.9,
            name=group)
        data_tab_2.append(trace)
    layout = go.Layout(
        xaxis={'title': 'Time'},
        yaxis={'title': 'Intensity'},
        margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
        hovermode='closest')

    fig = go.Figure(data=data_tab_2, layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=True)
    fig.update_layout(
        hovermode='closest', xaxis_title='Time',
        yaxis_title='Traffic Intensity',
        title_font_color='black',
        title_font_size=24,
        paper_bgcolor=colors['background_objects'],
        plot_bgcolor=colors['background_objects'])

    return fig, avg_speed2, avg_intensity2, mae2

@app.callback(
    Output('graph3', 'figure'),
    Output('avg_speed3', 'figure'),
    Output('avg_intensity3', 'figure'),
    Output('mae3', 'figure'),
    Input('area3', 'value'),
    Input('dates3', 'start_date'),
    Input('dates3', 'end_date'),
    Input('algorithms3', 'value'))
def update_tab_3(area, start_date, end_date, algorithm):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_dash_location = df_dash.loc[(df_dash['naam_meetlocatie_mst'] == area) & (pd.to_datetime(df_dash['start_meetperiode']) >= start_date) & (pd.to_datetime(df_dash['start_meetperiode']) <= end_date)]
    df_dash_location['start_meetperiode'] = pd.to_datetime(df_dash_location['start_meetperiode'])
    df_dash_location = df_dash_location[['start_meetperiode', 'intensiteit_anyVehicle']].set_index('start_meetperiode').asfreq('H').reset_index(drop=False)

    avg_speed3 = make_kpi(np.mean(df_dash_location['intensiteit_anyVehicle'])/10)
    avg_intensity3 = make_kpi(np.mean(df_dash_location['intensiteit_anyVehicle']))
    mae3 = make_kpi(np.mean(df_dash_location['intensiteit_anyVehicle'])/2)

    data_tab_3 = []
    area_name_prediction = location_mapping.loc[location_mapping['naam_meetlocatie_mst'] == area, 'id_meetlocatie'].item()
    if 'TBATS hourly' in algorithm:
        total_df, mae = tbats_predict_hourly(df_dash_location, area_name_prediction, end_date)
        data_tab_3.append(go.Scatter(
            x=total_df.start_meetperiode.tolist(), 
            y=total_df.prediction.tolist(),
            name=f'Predicted({area})',
            line=dict(color=colors['color_Amst'])))

    if 'TBATS daily' in algorithm:
        total_df, mae = tbats_predict_daily(df_dash_location, area_name_prediction, end_date)
        data_tab_3.append(go.Scatter(
            x=total_df.start_meetperiode.tolist(), 
            y=total_df.prediction.tolist(),
            name=f'Predicted({area})',
            line=dict(color=colors['color_Amst'])))

    data_tab_3.append(go.Scatter(
        x=total_df.start_meetperiode.tolist(),
        y=total_df.intensiteit_anyVehicle.tolist(),
        name='Actual',
        line=dict(dash='dot', color=colors['color_VU'])))

    layout = go.Layout(
        title=f'Selected location: <b>{area}</b>',
        xaxis={'title': 'Time'},
        yaxis={'title': 'Intensity'},
        margin={'l': 40, 'b': 40, 't': 80, 'r': 40},
        hovermode='closest')
    fig = go.Figure(data=data_tab_3, layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=True)
    fig.update_layout(
        hovermode='closest', xaxis_title='Time',
        yaxis_title='Traffic Intensity',
        title_font_color='black',
        title_font_size=24,
        paper_bgcolor=colors['background_objects'],
        plot_bgcolor=colors['background_objects'])

    return fig, avg_speed3, avg_intensity3, mae3

if __name__ == '__main__':
    app.run_server(debug=True)
