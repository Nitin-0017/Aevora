"""Fleet and historical engine views over verified demonstration data."""
import altair as alt
import pandas as pd
import streamlit as st
from .data import MODEL_NAMES, engine_history, risk_state

CYAN = '#59D6E8'


def chart(frame, x, y, *, color=None, points=False):
    plot = alt.Chart(frame).mark_circle(size=24, opacity=.35) if points else alt.Chart(frame).mark_line(strokeWidth=2)
    fields = dict(x=x, y=y, tooltip=list(frame.columns))
    if color:
        fields['color'] = color
    else:
        plot = plot.configure_mark(color=CYAN)
    st.altair_chart(plot.encode(**fields).properties(height=290), width='stretch')


def show_prediction(value, high, elevated):
    state = risk_state(value, high, elevated)
    display = state['display_prediction']
    st.metric('Raw estimated RUL', f'{display:.2f} cycles' if display is not None else 'Unavailable')
    if state['level'] in ('invalid', 'elevated'):
        st.warning(state['status'])
    elif state['level'] == 'high':
        st.error(state['status'])
    else:
        st.info(state['status'])
    st.caption('Raw output is not clipped. Negative RUL is a model failure, not physical remaining life. '
               'Illustrative bands are not failure probabilities or validated maintenance thresholds.')


def fleet_view(snapshot, fleet, model, cycle, high, elevated):
    st.subheader('Fleet overview')
    st.write('Explore estimated remaining life across held-out demonstration engines at a chosen historical cycle.')
    a,b,c = st.columns(3)
    a.metric('Available demo engines', snapshot.raw.unit_id.nunique())
    b.metric('Finite predictions', len(fleet))
    c.metric('Negative outputs · review required', int((fleet.predicted_rul < 0).sum()))
    st.caption(f'{MODEL_NAMES[model]} · replay through cycle {cycle} · uses each engine’s latest available reading at or before that cycle.')
    left,right=st.columns([1,1.4])
    with left:
        hist=alt.Chart(fleet).mark_bar(color=CYAN).encode(
            x=alt.X('predicted_rul:Q',bin=alt.Bin(maxbins=12),title='Raw predicted RUL (cycles)'),
            y=alt.Y('count():Q',title='Engines'),tooltip=['count()'])
        st.altair_chart(hist.properties(height=300),width='stretch')
    with right:
        table=fleet.rename(columns={'unit_id':'Engine','cycle':'Latest cycle','predicted_rul':'Raw RUL (cycles)'}).copy()
        table['Illustrative status']=[risk_state(v,high,elevated)['status'] for v in fleet.predicted_rul]
        st.dataframe(table, hide_index=True, width='stretch',column_config={'Raw RUL (cycles)':st.column_config.NumberColumn(format='%.2f')})
    st.info('This view contains the 20 validation engines, not the 80 training engines or official test engines. '
            'Complete source histories include failure endpoints; changing the replay cycle is historical exploration, not live monitoring.')


def engine_view(snapshot, fleet, model, cycle, high, elevated):
    st.subheader('Engine explorer')
    engine=st.selectbox('Engine', sorted(snapshot.raw.unit_id.unique().tolist()))
    history=engine_history(snapshot.raw,engine,cycle)
    row=fleet.loc[fleet.unit_id==engine].iloc[0]
    left,right=st.columns([1,2])
    with left:
        show_prediction(row.predicted_rul,high,elevated)
        st.write(f'**Engine {engine}** · {len(history)} available cycles')
        st.caption(f'Selected model: {MODEL_NAMES[model]} · inference verified')
    with right:
        sensors=st.multiselect('Sensors',snapshot.manifest['selected_sensors'],default=snapshot.manifest['selected_sensors'][:2])
        first,last=int(history.cycle.min()),int(history.cycle.max())
        selected=st.slider('Visible sensor cycle range',first,last,(first,last)) if last>first else (first,last)
        readings=history[history.cycle.between(*selected)]
        if sensors:
            long=readings.melt(id_vars='cycle',value_vars=sensors,var_name='Sensor',value_name='Reading')
            # Separate axes avoid misleading direct comparisons between sensor scales.
            plot=alt.Chart(long).mark_line(color=CYAN).encode(x=alt.X('cycle:Q',title='Operating cycle'),
                y=alt.Y('Reading:Q',title='Recorded value (dataset units)'),tooltip=['cycle','Reading','Sensor']).properties(height=150)
            st.altair_chart(plot.facet(row='Sensor:N').resolve_scale(y='independent'),width='stretch')
        else:
            st.info('Select at least one sensor to display its recorded history.')
    saved=snapshot.predictions[model]
    saved=saved[(saved.unit_id==engine)&(saved.cycle<=int(row.cycle))]
    st.markdown('**Saved prediction history**')
    st.caption('Actual stored validation predictions aligned by engine and endpoint cycle; no invented interpolation. Sensor range changes only the sensor chart.')
    if saved.empty:
        st.info('No saved eligible prediction history for this range.')
    else:
        chart(saved[['cycle','predicted_rul']],alt.X('cycle:Q',title='Endpoint cycle'),alt.Y('predicted_rul:Q',title='Raw predicted RUL (cycles)'))
    st.warning('Estimates may overstate remaining life. Complete validation histories and repeated model selection cannot demonstrate field reliability.')
