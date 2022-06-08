import streamlit as st
import config # Our config.py file with title, members and promotion

# For the different tabs we have
from collections import OrderedDict
from tabs import A_intro, B_data, C_dataviz, D_modelisation, E_simulation, F_conclusions

# For the age_encode class
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# Define the age_encoder class used in the model
class age_encoder(BaseEstimator, TransformerMixin):
    def __main__(self):
        super().__main__()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['age'] = pd.cut(X.age, bins=[0,12,20,40,60,99], labels=[1,2,3,4,5] )
        return X


#--------------------------------
# CONFIGURATION OF THE APP
#--------------------------------
st.set_page_config(
    page_title=config.TITLE,
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state = 'auto'
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

#--------------------------------
# TABS DEFINITION
#--------------------------------
TABS = OrderedDict(
    [
        (A_intro.sidebar_name, A_intro),
        (B_data.sidebar_name, B_data),
        (C_dataviz.sidebar_name, C_dataviz),
        (D_modelisation.sidebar_name, D_modelisation),
        (E_simulation.sidebar_name, E_simulation),
        (F_conclusions.sidebar_name, F_conclusions)
    ]
)

#--------------------------------
# CONTENT OF THE LEFT TAB
#--------------------------------
def run():
    st.sidebar.image('assets/logo_small_purple.png', width=300)

    # Radio buttons
    tab_name = st.sidebar.radio("Navigation", list(TABS.keys()), 0)

    st.sidebar.markdown("---")

    # Promotion
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("---")

    # Team members
    st.sidebar.markdown("## **Membres de l'équipe :**")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
