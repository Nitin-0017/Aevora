"""Streamlit composition and cache boundaries for offline inference."""
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st
from .data import (MODEL_NAMES, DEMO_LABEL, cache_token, load_snapshot, load_predictor,
                   fleet_predictions, ArtifactError, engine_history, infer_history)
from .views import fleet_view, engine_view
from .operator_view import machine_view
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
    st.sidebar.title('Your workspace')
    page=st.sidebar.radio('View',['Machine overview','Technical details'])
    model=st.sidebar.selectbox('Prediction model',list(MODEL_NAMES),format_func=MODEL_NAMES.get)
    st.sidebar.caption('Select an engine → run a prediction → review the suggested next step.')
    with st.sidebar.expander('About remaining life'):
        from .operator import RUL_EXPLANATION
        st.write(RUL_EXPLANATION)
        st.caption('Actual failure timing may differ.')
    with st.sidebar.expander('Demonstration thresholds'):
        high=st.number_input('Needs review at or below (cycles)',min_value=0.,value=30.,step=5.)
        elevated=st.number_input('Monitor at or below (cycles)',min_value=0.,value=100.,step=5.)
        st.caption('Illustrative, display-only categories. Not validated maintenance thresholds or safety statuses.')
    if high>=elevated:
        st.error('The review threshold must be below the monitoring threshold.')
        st.stop()
    try:
        token=cache_token(CONFIG)
        with st.spinner('Checking the saved models and demonstration data…'):
            snapshot=snapshot_resource(str(CONFIG),token)
        with st.sidebar.expander('Historical replay settings'):
            cycle=st.slider('Use recorded history through cycle',snapshot.settings.sequence_window,
                            int(snapshot.raw.cycle.max()),100)
            st.caption('This is recorded demonstration history, not live telemetry. Engines whose records end earlier use their final available cycle.')
        if page=='Machine overview':
            def run_prediction(engine):
                predictor=predictor_resource(str(CONFIG),token,model)
                history=engine_history(snapshot.raw,engine,cycle)
                row=infer_history(snapshot,model,predictor,history).iloc[0]
                return {'prediction':float(row.predicted_rul),
                        'generated_at':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
            machine_view(snapshot,model,cycle,high,elevated,token,run_prediction)
        else:
            section=st.selectbox('Technical section',['Fleet overview','Engine explorer','RUL prediction',
                                'Model comparison','Reliability insights','Model & data information'])
            if section in ('Fleet overview','Engine explorer','RUL prediction'):
                if st.button('Refresh technical inference'):
                    fleet_data.clear()
                with st.spinner('Running saved model inference…'):
                    fleet=fleet_data(str(CONFIG),token,model,cycle)
                if section=='Fleet overview':
                    fleet_view(snapshot,fleet,model,cycle,high,elevated)
                elif section=='Engine explorer':
                    engine_view(snapshot,fleet,model,cycle,high,elevated)
                else:
                    prediction_view(snapshot,fleet,model,cycle,high,elevated)
            elif section=='Model comparison':
                comparison_view(snapshot)
            elif section=='Reliability insights':
                reliability_view(snapshot)
            else:
                information_view(snapshot)
    except (ArtifactError,OSError,ValueError,KeyError,RuntimeError) as exc:
        st.error('Unable to assess: the demonstration data or saved models could not be verified. Restore the matching local files and try again.')
        with st.expander('Technical error details'):
            st.code(str(exc))
        st.info('See docs/dashboard.md for setup. No substitute predictions or automatic training will be used.')
