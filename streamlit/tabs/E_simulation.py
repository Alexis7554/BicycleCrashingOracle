import streamlit as st
import pandas as pd
import datetime
from joblib import load
import folium
from streamlit_folium import folium_static


title = "Simulation"
sidebar_name = "Simulation"

# Flag used to display the results only if the simulation has been run at least once
b_simulation_launched = False

# Variables to hold the results
gravity_A_prev = 0.0
gravity_A = 0.0

gravity_B_prev = 0.0
gravity_B = 0.0

nb_dangers_A_prev = 0
nb_dangers_A = 0

nb_dangers_B_prev = 0
nb_dangers_B = 0

coords_A = []
coords_B = []

#--------------------------------
# FUNCTION DEFINITION
#--------------------------------

# From the selected parameters, defines the dataset and computes the predictions
def set_sim_dataset(param_age, param_sexe, param_lum, param_atm, param_surf, param_traffic, param_dayofweek):

    raw_df_A = pd.read_csv("data/simulation_db_A.csv", sep=';')
    df_A = raw_df_A.drop(columns=["type_int", "c1", "c2"])

    raw_df_B = pd.read_csv("data/simulation_db_B.csv", sep=';')
    df_B = raw_df_B.drop(columns=["type_int", "c1", "c2"])

    for df in [df_A, df_B]:
        df.age = param_age

        df.sexe_male = 1 - param_sexe
        df.sexe_female = param_sexe

        #"Aube":0, "Pleine journée":1, "Nuit sans éclairage":2, "Nuit avec éclairage":3
        if param_lum == 0:
            df.lum_dawn_dusk = 1
        elif param_lum == 1:
            df.lum_day = 1
        elif param_lum == 2:
            df.lum_night_dark = 1
        elif param_lum == 3:
            df.lum_night_w_light = 1

        #"Normale":0, "Venteux":1, "Brouillard":2, "Pluie":3, "Neige":4
        if param_atm == 0:
            df.atm_normal = 1
        elif param_atm == 1:
            df.atm_wind_storm = 1
        elif param_atm == 2:
            df.atm_fog_smoke = 1
        elif param_atm == 3:
            df.atm_rain = 1
        elif param_atm == 4:
            df.atm_snow_hail = 1

        #"Normale":0, "Venteux":0, "Brouillard":0, "Pluie":1, "Neige":2
        if param_surf == 0:
            df.surf_normal = 1
        elif param_surf == 1:
            df.surf_wet = 1
        elif param_surf == 2:
            df.surf_icy = 1

        df.heavy_traffic_0 = 1 - param_traffic
        df.heavy_traffic_1 = param_traffic

        #"Lundi":0, "Mardi":1, "Mercredi":2, "Jeudi":3, 'Vendredi':4, 'Samedi':5, 'Dimanche':6
        if param_dayofweek == 0:
            df.dayofweek_0 = 1
        elif param_dayofweek == 1:
            df.dayofweek_1 = 1
        elif param_dayofweek == 2:
            df.dayofweek_2 = 1
        elif param_dayofweek == 3:
            df.dayofweek_3 = 1
        elif param_dayofweek == 4:
            df.dayofweek_4 = 1
        elif param_dayofweek == 5:
            df.dayofweek_5 = 1
        elif param_dayofweek == 6:
            df.dayofweek_6 = 1
        

    #clf= load('Full_Model.pkl')
    scaler= load('scaler.pkl')
    clf= load('model_tni.pkl')

    df_A = scaler.transform(df_A)
    df_B = scaler.transform(df_B)

    global gravity_A, gravity_A_prev
    gravity_A_prev = gravity_A
    preds_A = clf.predict(df_A)
    gravity_A = preds_A.mean()

    global gravity_B, gravity_B_prev
    gravity_B_prev = gravity_B
    preds_B = clf.predict(df_B)
    gravity_B = preds_B.mean()

    global nb_dangers_A, nb_dangers_A_prev
    nb_dangers_A_prev = nb_dangers_A
    nb_dangers_A = preds_A.sum()

    global nb_dangers_B, nb_dangers_B_prev
    nb_dangers_B_prev = nb_dangers_B
    nb_dangers_B = preds_B.sum()

    # Handle coordinates
    global coords_A
    coords_A = raw_df_A.iloc[:,-3:]
    coords_A['grav'] = preds_A

    global coords_B
    coords_B = raw_df_B.iloc[:,-3:]
    coords_B['grav'] = preds_B


# Adds markers on a folium map, to display the results
def add_markers_on_map(folium_map, coords):

    display_colors = ["green", "red"]

    coord_points = coords[["c1","grav"]][coords.type_int == 0].values
    coord_streets = coords[["c1","c2","grav"]][coords.type_int == 1].values

    # We need to change the list for the streets, so folium can display it (list of list of floats)
    coord_streets_clean = []
    for i in range(len(coord_streets)):
        pt1 = [float(i) for i in coord_streets[i, 0].split(", ")]
        pt2 = [float(i) for i in coord_streets[i, 1].split(", ")]
        coord_streets_clean.append([[pt1, pt2], coord_streets[i, 2]])

    # Add the lines for the streets
    for i in range(len(coord_streets_clean)):
        folium.PolyLine(coord_streets_clean[i][0],
                color=display_colors[coord_streets_clean[i][1]],
                weight=5,
                opacity=0.8
                ).add_to(folium_map)

    # Add the points pour the intersections
    for i in range(len(coord_points)):
        folium.Circle(
            location=coord_points[i,0].split(', '),
            radius=5,
            color=display_colors[coord_points[i,1]],
            fill=True,
            fillcolor = display_colors[coord_points[i,1]],
            fill_opacity=0.8,
            opacity=0.0
            ).add_to(folium_map)


#--------------------------------
# CONTENT OF THE PAGE
#--------------------------------
#@st.cache(suppress_st_warning=True)
def run():
    
    st.title(title)

    st.write("Dans cette section, vous allez pouvoir choisir les paramètres d'une simulation pour 2 trajets dans Paris.\n"
             "En fonction de ces paramètres, notre modèle déterminera la gravité probable d'un accident sur chaque étape de chaque trajet.")

    col1, col2, col3 = st.columns([10, 1, 10])

    with col1:
        st.subheader('Itinéraire A')

        st.markdown("Ici nous passons autour du rond-point de l'Etoile via des rues parallèles, puis par l'avenue Foch, la porte Dauphine et enfin dans le Bois de Boulogne.")
        st.markdown("Temps estimé : 9 minutes.")

        st.image('assets/trajet2.PNG', width=600)

        with open('data/simulation_db_A.csv') as f:
            st.download_button('CSV du trajet A', f, file_name='simulation_db_A.csv')

    with col3:
        st.subheader('Itinéraire B')

        st.markdown("Ici nous passons sous le rond-point de l'Etoile via un tunnel souterrain, puis par le rond-point de la Porte Maillot et enfin dans le Bois de Boulogne.")
        st.markdown("Temps estimé : 11 minutes")

        st.image('assets/trajet1.PNG', width=600)

        with open('data/simulation_db_B.csv') as f:
            st.download_button('CSV du trajet B', f, file_name='simulation_db_B.csv')

    st.markdown("***")
    st.header('Paramétrage')

    with st.form("params"):
        # age
        slider_age = st.slider("Quel est votre âge", min_value=5, max_value=80, value=30)
        
        # atm & surf
        dict_atm = {"Normale":0, "Venteux":1, "Brouillard":2, "Pluie":3, "Neige":4}
        dict_surf = {"Normale":0, "Venteux":0, "Brouillard":0, "Pluie":1, "Neige":2}
        list_weather = st.selectbox(
            'Comment est la météo ?',
            (dict_atm),
            index = 0)

        # lum
        dict_lum = {"Aube":0, "Pleine journée":1, "Nuit sans éclairage":2, "Nuit avec éclairage":3}
        list_lum = st.selectbox(
            'Quelle est la luminosité ambiante ?',
            (dict_lum),
            index = 1)

        # dayofweek
        dict_dayofweek = {"Lundi":0, "Mardi":1, "Mercredi":2, "Jeudi":3, 'Vendredi':4, 'Samedi':5, 'Dimanche':6}
        list_dayofweek = st.selectbox(
            'Quel jour sommes-nous ?',
            (dict_dayofweek),
            index = 0)

        # heavytraffic
        timeinput = st.time_input("Quelle heure est-il ?", value=datetime.time(8, 00))
        heavytraffic = 0
        if ((7 <= timeinput.hour) & (timeinput.hour < 9)) | ((16 <= timeinput.hour) & (timeinput.hour < 19)):
            heavytraffic = 1

        # sexe
        dict_sexe = {"Homme":0, "Femme":1}
        list_sexe = st.selectbox(
            'Êtes-vous un homme ou une femme ?',
            (dict_sexe),
            index = 0)

        # Submit button
        submitted = st.form_submit_button("Lancer la simulation")
        if submitted:
            # We retrieve the selected values and transform them to int
            param_atm = dict_atm[list_weather]
            param_surf = dict_surf[list_weather]
            param_lum = dict_lum[list_lum]
            param_age = slider_age
            param_sexe = dict_sexe[list_sexe]
            param_traffic = heavytraffic
            param_dayofweek = dict_dayofweek[list_dayofweek]

            # We set this flag to True, so we can display the results
            global b_simulation_launched
            b_simulation_launched = True

            # We set our values to our dataset to run the simulation
            set_sim_dataset(param_age, param_sexe, param_lum, param_atm, param_surf,  param_traffic, param_dayofweek)

    st.markdown("***")

    st.header('Résultats')

    if not b_simulation_launched:
        st.info("Veuillez d'abord choisir vos paramètres et cliquer sur le bouton 'Lancer la simulation'.")
    else:

        # Common parameters for the 2 maps
        marker1 = "Charles De Gaulle Etoile"
        marker2 = "Fondation LOUIS VUITTON"

        # Initialize the maps
        mA = folium.Map(location=[48.875453652573306, 2.2827294884566167], tiles='https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png ',
                attr="BicycleCrashingOracle", zoom_start = 14)
        mB = folium.Map(location=[48.875453652573306, 2.2827294884566167], tiles='https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png ',
                attr="BicycleCrashingOracle", zoom_start = 14)
        
        # Add markers for start and stop
        # Start
        folium.Marker(
            [48.872963339184146, 2.2989970047304733], popup=marker1, tooltip=marker1,
        ).add_to(mA)
        folium.Marker(
            [48.872963339184146, 2.2989970047304733], popup=marker1, tooltip=marker1
        ).add_to(mB)
        # Stop
        folium.Marker(
            [48.876737153105324, 2.263765853391789], popup=marker2, tooltip=marker2, icon=folium.Icon(icon="flag")
        ).add_to(mA)
        folium.Marker(
            [48.876737153105324, 2.263765853391789], popup=marker2, tooltip=marker2, icon=folium.Icon(icon="flag")
        ).add_to(mB)

        # Display the 2 subheaders in two columns
        titlecol1, titlecol2, titlecol3 = st.columns([10, 1, 10])
        with titlecol1:
            st.subheader('Itinéraire A')

        with titlecol3:
            st.subheader('Itinéraire B')


        # Display the metrics, 2 for each route (so 4, in 5 columns)
        metriccol1,  metriccol2, metriccol3, metriccol4,  metriccol5= st.columns([10, 10, 2, 10, 10])

        # Mean danger for A
        with metriccol1:
            value_A = f'{gravity_A:.2f}'
            delta_A = f'{gravity_A - gravity_A_prev:.2f}'
            delta_color_A ="inverse"
            if delta_A == "0.00":
                delta_color_A ='off'
            st.metric(label="Dangerosité moyenne (entre 0 et 1)", value=value_A, delta=delta_A, delta_color=delta_color_A)
        
        # Number of dangerous points for A
        with metriccol2:
            value_A = f'{nb_dangers_A}'
            delta_A = f'{nb_dangers_A - nb_dangers_A_prev}'
            delta_color_A ="inverse"
            if delta_A == "0":
                delta_color_A ='off'
            st.metric(label="Nombre de zones dangereuses", value=value_A, delta=delta_A, delta_color=delta_color_A)

        # Mean danger for B
        with metriccol4:
            value_B = f'{gravity_B:.2f}'
            delta_B = f'{gravity_B - gravity_B_prev:.2f}'
            delta_color_B="inverse"
            if delta_B == "0.00":
                delta_color_B='off'
            st.metric(label="Dangerosité moyenne (entre 0 et 1)", value=value_B, delta=delta_B, delta_color=delta_color_B)
        
        # Number of dangerous points for B
        with metriccol5:
            value_B = f'{nb_dangers_B}'
            delta_B = f'{nb_dangers_B - nb_dangers_B_prev}'
            delta_color_B="inverse"
            if delta_B == "0":
                delta_color_B='off'
            st.metric(label="Nombre de zones dangereuses", value=value_B, delta=delta_B, delta_color=delta_color_B)


        # Display the 2 maps in two columns, with a separator
        mapcol1, mapcol2, mapcol3 = st.columns([10, 1, 10])
        with mapcol1:
            # Get the coordinates and use them to add markers
            add_markers_on_map(mA, coords_A)

            # Render the map in Streamlit
            folium_static(mA)

        with mapcol3:
            # Get the coordinates and use them to add markers
            add_markers_on_map(mB, coords_B)

            # Render the map in Streamlit
            folium_static(mB)