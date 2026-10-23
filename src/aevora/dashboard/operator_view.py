"""Operator-first view using real predictions and recorded sensor readings."""
import altair as alt
import streamlit as st
from .data import MODEL_NAMES, engine_history
from .operator import operator_summary, operating_hours, RUL_EXPLANATION, GUIDANCE_LABEL


def machine_view(snapshot, model, cycle, high, elevated, token, run_prediction):
    st.subheader('Your demonstration engine')
    with st.expander('How to use AEVORA'):
        st.markdown('1. Select a demonstration engine.\n2. Select a supported model in the sidebar.\n3. Run a prediction.\n4. Read the estimated remaining life and suggested next step.\n5. Review the limitations before making maintenance decisions.')
        st.write(RUL_EXPLANATION)
        st.write('Actual failure timing may differ.')
    left,right = st.columns([3,1],vertical_alignment='bottom')
    with left:
        engine=st.selectbox('Select a demonstration engine',sorted(snapshot.raw.unit_id.unique().tolist()),
                            format_func=lambda value:f'Engine {value}',key='operator_engine')
    with right:
        run=st.button('Run prediction',type='primary',width='stretch')
    history=engine_history(snapshot.raw,engine,cycle)
    identity=(token,model,engine,cycle)
    if run:
        st.session_state.pop('operator_result',None)
        try:
            with st.spinner('Reading recorded sensor history and estimating remaining life…'):
                result=run_prediction(engine)
            st.session_state.operator_result={'identity':identity,**result}
        except (OSError,ValueError,RuntimeError) as exc:
            st.session_state.operator_result={'identity':identity,'prediction':None,'error':str(exc)}
    stored=st.session_state.get('operator_result')
    result=stored if stored and stored['identity']==identity else None
    value=result['prediction'] if result else None
    summary=operator_summary(value,high,elevated)
    with st.container(border=True):
        st.markdown(f'### Engine {engine} · {summary["icon"]} {summary["status"]}')
        a,b=st.columns([1,1.5])
        with a:
            st.metric('Estimated remaining life',summary['formatted_cycles'] + (' cycles' if summary['display_prediction'] is not None else ''))
            st.caption('Operating cycles, not hours or days. Actual failure timing may differ.')
        with b:
            st.markdown('**Recommended next step**')
            st.write(summary['action'] if result else 'Select your engine and model, then choose Run prediction. No estimate is available for these selections yet.')
            st.caption(GUIDANCE_LABEL)
        st.caption(f'Latest available operating cycle: {int(history.cycle.max())} · Model: {MODEL_NAMES[model]}')
        if result and result.get('generated_at'):
            st.caption(f'Prediction generated: {result["generated_at"]} · application time, not sensor measurement time')
        st.caption('NASA C-MAPSS demonstration data — not a live industrial machine. These are illustrative categories, not validated safety statuses.')
    if result and result.get('error'):
        st.error('Unable to assess this engine. Check the recorded inputs and local model files before trying again.')
        with st.expander('Technical error details'):
            st.code(result['error'])
    elif result and summary['level']=='invalid':
        st.warning(f'Invalid model output: {value!r}. It has not been clipped or replaced with a reassuring estimate.')
    st.markdown('**Why are we showing this?**')
    if result:
        st.write(summary['reason'])
    st.write('AEVORA analyzes historical sensor readings and estimates remaining life from patterns learned in training data. '
             'It can be wrong. This is not a failure guarantee or a substitute for inspections. '
             'A concerning estimate should prompt inspection and review, not automatic shutdown unless a separately validated safety procedure requires it.')
    st.caption('No causal explanation or sensor importance is claimed for this prediction.')
    with st.expander('How reliable is this estimate?'):
        score=snapshot.metrics.set_index('model').loc[model]
        st.write(f'In development checks, {MODEL_NAMES[model]} differed from actual remaining life by '
                 f'{score.mae:.2f} cycles on average; its largest observed absolute error was {score.max_absolute_error:.2f} cycles.')
        st.write('Those checks helped choose the models, so they may look better than results on unseen engines. '
                 'They do not provide a confidence percentage or a guaranteed error range for this engine. '
                 'Use qualified inspection and approved procedures before making maintenance decisions.')
        st.caption('Development validation only · same 3,472 observations / 20 engines. See Technical details for complete metrics.')
    with st.expander('Optional: estimate operating hours'):
        verified=st.checkbox('I have a verified average operating rate',key='verified_rate')
        if verified:
            rate=st.number_input('Verified operating rate (cycles per hour)',min_value=0.,value=None,
                                 placeholder='Enter your measured rate',key='operating_rate')
            try:
                hours=operating_hours(value,rate)
                if hours is not None:
                    st.write(f'Approximately **{hours:,.2f} operating hours** at {rate:g} cycles per hour.')
                    st.caption('Assumption-based estimate at a constant average rate, not a guaranteed failure time. No failure date is calculated.')
                else:
                    st.info('Enter your verified rate and run a valid prediction to calculate operating hours.')
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.caption('No hours or days estimate is shown without a verified rate supplied by you.')
    st.markdown('**Recorded sensor trend**')
    sensor=st.selectbox('Sensor reading',snapshot.manifest['selected_sensors'],
                        format_func=lambda name:'Sensor '+name.removeprefix('sensor_'),key='operator_sensor')
    chart=alt.Chart(history).mark_line(color='#59D6E8').encode(
        x=alt.X('cycle:Q',title='Operating cycle'),
        y=alt.Y(f'{sensor}:Q',title='Recorded reading (dataset units)',scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('cycle:Q',title='Cycle'),alt.Tooltip(f'{sensor}:Q',title='Reading')]).properties(height=180)
    st.altair_chart(chart,width='stretch')
    st.caption('Real recorded values through the selected replay cycle. This chart alone does not identify a fault or explain the model’s decision.')
    with st.expander('Technical prediction details'):
        st.write('Raw model output (cycles):',repr(value) if result else 'No prediction run')
        st.write('The main card rounds valid values to two decimal places; no clipping is applied. Negative values are invalid and remain disclosed above.')
        st.caption('Frozen preprocessing v1 · 20 timesteps × 75 features · classical summary: 225 features')
