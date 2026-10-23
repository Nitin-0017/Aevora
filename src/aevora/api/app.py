"""Start with: uvicorn aevora.api.app:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import logging
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .service import Service, DISCLOSURE
from ..dashboard.data import MODEL_NAMES, ArtifactError

ModelID = Literal['baseline','ridge','gradient_boosting','lstm']
CONFIG = Path(__file__).resolve().parents[3] / 'configs/default.toml'


class FleetRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    model: ModelID = 'gradient_boosting'
    cycle: int = Field(default=100, ge=20, le=10000)
    review_threshold: float = Field(default=30., ge=0, le=100000)
    monitor_threshold: float = Field(default=100., gt=0, le=100000)

    @model_validator(mode='after')
    def ordered_thresholds(self):
        if self.review_threshold >= self.monitor_threshold:
            raise ValueError('Review threshold must be below monitor threshold.')
        return self


class PredictRequest(FleetRequest):
    engine_id: int = Field(gt=0)


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    engine_id: int
    model: ModelID
    raw_rul: float
    valid: bool
    status: str
    level: str
    recommendation: str
    reason: str
    latest_cycle: int
    available_cycles: int
    preprocessing_version: int
    model_version: int
    generated_at: str
    warning: str
    guidance: str


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    generated_at: str
    disclosure: str


def error(status, code, message):
    return JSONResponse(status_code=status,content={'error':{'code':code,'message':message}})


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(application):
        try:
            application.state.service = service if service is not None else Service(CONFIG)
        except Exception:
            logging.getLogger(__name__).exception('Local artifact initialization failed')
            application.state.service = None
        yield

    app = FastAPI(title='AEVORA local inference API',version='1.0',lifespan=lifespan)
    app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],
                       allow_methods=['GET','POST'],allow_headers=['Content-Type'],allow_credentials=False)

    def current():
        active = app.state.service
        if active is None:
            raise ArtifactError('Restore the matching local dataset, model and audit artifacts; then restart the API.')
        active.check()
        return active

    @app.exception_handler(ArtifactError)
    async def artifact_error(request:Request, exc:ArtifactError):
        return error(503,'artifacts_unavailable','Saved models or data could not be verified. Restore the audited local bundle and restart the API.')

    @app.exception_handler(RequestValidationError)
    async def validation_error(request:Request, exc:RequestValidationError):
        return error(422,'invalid_request','Check engine, model, cycle and thresholds. Use a supported model, cycle ≥20, and review threshold < monitor threshold.')

    @app.exception_handler(ValueError)
    async def value_error(request:Request, exc:ValueError):
        return error(422,'unable_to_assess','Unable to assess. Check the selected engine, sensor and input history. Invalid model outputs must not guide maintenance.')

    @app.exception_handler(Exception)
    async def unexpected_error(request:Request, exc:Exception):
        logging.getLogger(__name__).exception('Request failed')
        return error(503,'inference_unavailable','Inference is unavailable. Check the saved model bundle and restart the API.')

    @app.get('/api/health')
    def health():
        try:
            current()
        except ArtifactError:
            return JSONResponse(status_code=503,content={'status':'unavailable','artifacts_ready':False,'disclosure':DISCLOSURE})
        return {'status':'ready','artifacts_ready':True,'disclosure':DISCLOSURE}

    @app.get('/api/engines')
    def engines():
        s = current()
        return {'engines':s.engines(),'sensors':s.snapshot.manifest['selected_sensors'],
                'min_cycle':s.snapshot.settings.sequence_window,'max_cycle':int(s.snapshot.raw.cycle.max()),'disclosure':DISCLOSURE}

    @app.get('/api/engines/{engine_id}/history')
    def history(engine_id:int, cycle:int=Query(100,ge=20,le=10000), sensor:str='sensor_2'):
        return current().history(engine_id,cycle,sensor)

    @app.get('/api/models')
    def models():
        s=current()
        return {'models':[{'id':key,'name':value,'schema_version':1,'preprocessing_version':s.snapshot.manifest['version'],
                           'sequence_length':s.snapshot.settings.sequence_window,'feature_count':len(s.snapshot.manifest['feature_names'])}
                          for key,value in MODEL_NAMES.items()]}

    @app.post('/api/predict',response_model=PredictionResponse)
    def predict(body:PredictRequest):
        return current().predict(body)

    @app.post('/api/fleet-predictions',response_model=PredictionResponse)
    def fleet(body:FleetRequest):
        return current().predict(body,fleet=True)

    @app.get('/api/model-metrics')
    def metrics():
        return {'scope':'Development validation used for model selection; not unbiased final generalization.',
                'metrics':current().snapshot.metrics.to_dict('records')}

    @app.get('/api/diagnostics/{model}')
    def diagnostics(model:ModelID):
        return current().diagnostics(model)

    @app.get('/api/reliability')
    def reliability():
        return current().reliability()

    return app


app = create_app()
