"""Development-validation comparison and reliability presentation."""
import altair as alt
import pandas as pd
import streamlit as st
from .data import MODEL_NAMES, VALIDATION_LABEL, engine_history
from .views import show_prediction


def comparison_view(snapshot):
    st.subheader('Model comparison')
    st.caption(VALIDATION_LABEL)
    st.warning('Validation selected hyperparameters and the LSTM checkpoint. These are development results, not unbiased final generalization estimates.')
    names={'model':'Model','mae':'MAE (cycles)','rmse':'RMSE (cycles)','r2':'R²',
           'within_10_pct':'Within ±10 cycles (%)','within_20_pct':'Within ±20 cycles (%)',
           'mean_signed_error':'Mean bias (cycles)','median_absolute_error':'Median absolute error (cycles)',
           'p95_absolute_error':'P95 absolute error (cycles)','max_absolute_error':'Maximum absolute error (cycles)'}
    table=snapshot.metrics[list(names)].rename(columns=names).copy()
    table['Model']=table.Model.map(MODEL_NAMES)
    st.dataframe(table,hide_index=True,width='stretch')
    st.caption('Bias = prediction − actual RUL; positive means remaining life is overstated. Tolerance percentages are regression measures, not classification accuracy. P95 is an error quantile, not a confidence interval.')
    st.altair_chart(alt.Chart(table).mark_bar(color='#59D6E8').encode(
        x=alt.X('Model:N',sort=None),y=alt.Y('MAE (cycles):Q'),tooltip=list(table.columns)).properties(height=260),width='stretch')
    chosen=st.selectbox('Diagnostic model',list(MODEL_NAMES),format_func=MODEL_NAMES.get)
    frame=snapshot.predictions[chosen].copy()
    frame['Residual (cycles)']=frame.predicted_rul-frame.rul
    a,b=st.columns(2)
    with a:
        st.markdown('**Actual versus predicted · n=3,472**')
        points=alt.Chart(frame).mark_circle(size=12,opacity=.25,color='#59D6E8').encode(
            x=alt.X('rul:Q',title='Actual RUL (cycles)'),y=alt.Y('predicted_rul:Q',title='Predicted RUL (cycles)'),
            tooltip=['unit_id','cycle','rul','predicted_rul'])
        lo=min(0.,frame.predicted_rul.min()); hi=max(frame.rul.max(),frame.predicted_rul.max())
        ideal=alt.Chart(pd.DataFrame({'rul':[lo,hi],'predicted_rul':[lo,hi]})).mark_line(color='#FBBF24',strokeDash=[5,5]).encode(x='rul:Q',y='predicted_rul:Q')
        st.altair_chart((points+ideal).properties(height=300),width='stretch')
        st.caption('Dashed amber line: ideal prediction (predicted = actual).')
    with b:
        st.markdown('**Residual distribution · n=3,472**')
        hist=alt.Chart(frame).mark_bar(color='#59D6E8').encode(
            x=alt.X('Residual (cycles):Q',bin=alt.Bin(maxbins=40)),y=alt.Y('count():Q',title='Observations'))
        zero=alt.Chart(pd.DataFrame({'Residual (cycles)':[0]})).mark_rule(color='#FBBF24').encode(x='Residual (cycles):Q')
        st.altair_chart((hist+zero).properties(height=300),width='stretch')
    st.download_button('Download verified comparison CSV',snapshot.metrics.to_csv(index=False),file_name='aevora_development_metrics.csv',mime='text/csv')


def reliability_view(snapshot):
    st.subheader('Reliability insights')
    st.caption(VALIDATION_LABEL)
    st.write('Average error conceals concentrated failures. The findings below describe this validation sample only.')
    frame=snapshot.predictions['gradient_boosting']
    residual=frame.predicted_rul-frame.rul
    middle=residual[(frame.rul>30)&(frame.rul<=100)]
    row=snapshot.metrics.set_index('model').loc['gradient_boosting']
    a,b,c,d=st.columns(4)
    a.metric('Boosted · middle-life bias',f'{middle.mean():+.2f} cycles')
    b.metric('Boosted · overall bias',f'{residual.mean():+.2f} cycles')
    c.metric('Boosted · largest overprediction',f'{residual.max():.2f} cycles')
    d.metric('Boosted · P95 absolute error',f'{row.p95_absolute_error:.2f} cycles')
    st.warning('Overprediction can delay inspection. The middle-life band is actual RUL (30,100] cycles; it is known retrospectively, not an online sensor reading.')
    negatives=pd.DataFrame([{'Model':MODEL_NAMES[name],'Negative predictions':int((f.predicted_rul<0).sum()),
                            'Observations':len(f)} for name,f in snapshot.predictions.items()])
    st.dataframe(negatives,hide_index=True,width='stretch')
    st.caption('Negative RUL is physically invalid and remains visible in errors. Finite output alone does not establish reliability.')
    st.markdown('**Synthetic sensor-noise sensitivity**')
    probe=snapshot.reliability['perturbation']
    st.write(f"Saved Phase 6 probe: independent zero-mean Gaussian noise, {100*probe['fraction']:.0f}% of frozen training sensor standard deviations, seed {probe['seed']}.")
    table=pd.DataFrame(snapshot.reliability['tables']['stability'])
    table['model']=table.model.map(MODEL_NAMES)
    table=table[['model','mean_absolute_prediction_change','p95_absolute_prediction_change','max_absolute_prediction_change']]
    table.columns=['Model','Mean absolute shift (cycles)','P95 absolute shift (cycles)','Maximum absolute shift (cycles)']
    st.dataframe(table,hide_index=True,width='stretch')
    st.info('This is a saved sensitivity experiment, not calibrated uncertainty, failure probability, or a realistic field fault test. It is not rerun by the dashboard.')
    st.write('Passing software tests verifies implementation behavior. It does not establish that model outputs are safe for maintenance decisions.')


def prediction_view(snapshot,fleet,model,cycle,high,elevated):
    st.subheader('RUL prediction')
    st.write('Choose a recorded validation engine. Use “Run / refresh inference” in the sidebar to rerun the saved model at the selected replay cycle.')
    engine=st.selectbox('Prediction engine',sorted(snapshot.raw.unit_id.unique().tolist()))
    row=fleet.loc[fleet.unit_id==engine].iloc[0]
    history=engine_history(snapshot.raw,engine,cycle)
    show_prediction(row.predicted_rul,high,elevated)
    st.success(f'Inference completed · engine {engine} · endpoint cycle {int(row.cycle)} · {len(history)} recorded cycles available')
    st.write(f'**Model:** {MODEL_NAMES[model]} · **Model schema:** v1 · **Preprocessing:** v{snapshot.manifest["version"]}')
    st.caption('20 chronological timesteps × 75 features. Classical models summarize these to 225 historical features. No scaler is fitted during inference.')
    st.write('RUL estimates remaining operating cycles, not a guaranteed failure date or elapsed hours. This selector supports validated local demonstration histories only; arbitrary uploads are not supported.')
    with st.expander('Feature order and provenance'):
        st.code('\n'.join(snapshot.manifest['feature_names']))
        st.write('Training source SHA-256:',snapshot.manifest['source_sha256'])
        st.json(snapshot.metadata[model]['model_config'])


def information_view(snapshot):
    st.subheader('Model & data information')
    st.write('AEVORA estimates remaining useful life from simulated turbofan sensor histories in NASA C-MAPSS FD001. This is an offline research demonstration, not live industrial monitoring.')
    st.markdown('''
- **Training engines (80):** used to select sensors, fit preprocessing and learn model parameters.
- **Validation engines (20):** withheld from fitting, but used for model selection and LSTM early stopping. These are the dashboard’s demonstration engines.
- **Official test engines and targets:** reserved for later evaluation. This dashboard does not load either file. Test targets remain unused to preserve a future evaluation with frozen choices.

Every estimate uses the frozen preprocessing and a saved model. The median baseline ignores sensor variation; Ridge is linear and can predict negative RUL; boosted trees have the best overall development errors but large overprediction cases; the LSTM has higher average error and substantial engine-to-engine variation.

**Not production-ready:** one dataset subset and split, correlated windows, selection-biased validation, negative outputs, large tail errors, no calibrated prediction intervals, and no field validation. Illustrative risk bands are not maintenance instructions. No autonomous action is taken.

A later approved evaluation should freeze model and eligibility choices before opening official test labels, use one final available endpoint per engine, and report it separately from today’s all-window validation metrics.
''')
    st.caption(f'Local artifact root: {snapshot.settings.artifact_dir}. Dependencies, raw training histories, saved models and both audit reports must already be present.')
