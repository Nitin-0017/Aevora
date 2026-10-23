"""Streamlit composition and cache boundaries for offline inference."""
from pathlib import Path
import streamlit as st
from .data import (MODEL_NAMES, DEMO_LABEL, cache_token, load_snapshot, load_predictor,
                   fleet_predictions, ArtifactError)
from .views import fleet_view, engine_view
from .reports import comparison_view, reliability_view, prediction_view, information_view

CONFIG = Path(__file__).resolve().parents[3] / 'configs/default.toml'


@st.cache_resource(show_spinner=False)
def snapshot_resource(config, token):
    return load_snapshot(config)


@st.cache_resource(show_spinner=False)
def predictor_resource(config, token, model):
    return load_predictor(snapshot_resource(config,token),model)


@st.cache_data(show_spinner=False)
def fleet_data(config,token,model,cycle):
    snapshot=snapshot_resource(config,token)
    return fleet_predictions(snapshot,model,predictor_resource(config,token,model),cycle)


def main():
    st.set_page_config(page_title='AEVORA · Intelligence Before Failure',page_icon='◈',layout='wide')
    st.markdown('''<style>
    .block-container {max-width:1500px;padding-top:2rem;}
    h1 {letter-spacing:.08em;} [data-testid="stMetric"] {border:1px solid #334155;border-radius:8px;padding:16px;}
    [data-testid="stSidebar"] {border-right:1px solid #334155;}
    </style>''',unsafe_allow_html=True)
    st.title('AEVORA')
    st.markdown('**Intelligence Before Failure.**')
    st.caption(DEMO_LABEL)
    st.sidebar.title('Operations workspace')
    page=st.sidebar.radio('View',['Fleet overview','Engine explorer','RUL prediction','Model comparison','Reliability insights','Model & data information'])
    model=st.sidebar.selectbox('Saved model',list(MODEL_NAMES),format_func=MODEL_NAMES.get)
    with st.sidebar.expander('Illustrative risk thresholds'):
        high=st.number_input('High concern at or below (cycles)',min_value=0.,value=30.,step=5.)
        elevated=st.number_input('Elevated concern at or below (cycles)',min_value=0.,value=100.,step=5.)
        st.caption('Display-only demonstration settings. No model parameter or saved configuration is changed.')
    if high>=elevated:
        st.error('High-concern threshold must be below the elevated threshold.')
        st.stop()
    try:
        token=cache_token(CONFIG)
        with st.spinner('Verifying local model, preprocessing and audit artifacts…'):
            snapshot=snapshot_resource(str(CONFIG),token)
        cycle=st.sidebar.slider('Replay through operating cycle',snapshot.settings.sequence_window,
                                int(snapshot.raw.cycle.max()),100)
        if st.sidebar.button('Run / refresh inference',type='primary'):
            fleet_data.clear()
        with st.spinner('Running saved model inference…'):
            fleet=fleet_data(str(CONFIG),token,model,cycle)
        st.sidebar.caption('Frozen preprocessing v1 · 20 timesteps × 75 ordered features · CPU inference')
        if page=='Fleet overview':
            fleet_view(snapshot,fleet,model,cycle,high,elevated)
        elif page=='Engine explorer':
            engine_view(snapshot,fleet,model,cycle,high,elevated)
        elif page=='RUL prediction':
            prediction_view(snapshot,fleet,model,cycle,high,elevated)
        elif page=='Model comparison':
            comparison_view(snapshot)
        elif page=='Reliability insights':
            reliability_view(snapshot)
        else:
            information_view(snapshot)
    except (ArtifactError,OSError,ValueError,KeyError,RuntimeError) as exc:
        st.error(str(exc))
        st.info('Check the local setup in docs/dashboard.md. The application will not download data, train models or substitute placeholder predictions.')
