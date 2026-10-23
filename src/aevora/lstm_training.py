"""Reproducible CPU training over verified Phase 3 sequences."""

import argparse
from copy import deepcopy
import importlib.metadata
import json
import platform
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .config import load_config
from .lstm import RULLSTM, LSTMPredictor
from .training import digest, load_prepared, metrics


class EarlyStopping:
    """Keep an independent copy of the best weights; ties count as no improvement."""
    def __init__(self, patience, min_delta=0.):
        if patience < 1 or min_delta < 0:
            raise ValueError("Invalid early-stopping controls")
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.best_epoch, self.bad_epochs = float('inf'), 0, 0
        self.state = None

    def update(self, loss, model, epoch):
        if not np.isfinite(loss):
            raise ValueError("Nonfinite validation loss")
        if loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.bad_epochs = float(loss), epoch, 0
            self.state = deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.state is None:
            raise ValueError("No best checkpoint exists")
        model.load_state_dict(self.state)


def fit_lstm(train, validation, features, config, seed):
    """Batches must come from load_prepared; additionally reject engine overlap here."""
    if set(train.endpoints.unit_id) & set(validation.endpoints.unit_id):
        raise ValueError("Training and validation engines overlap")
    if config['device'] != 'cpu':
        raise ValueError("This reproducible training path requires CPU")
    for batch in (train, validation):
        if (batch.x.ndim != 3 or batch.x.shape[2] != len(features) or not len(batch.x)
                or batch.y is None or batch.y.shape != (len(batch.x),)
                or len(batch.endpoints) != len(batch.x)
                or not np.isfinite(batch.x).all() or not np.isfinite(batch.y).all()):
            raise ValueError("Invalid aligned sequence/target arrays")
    if train.x.shape[1:] != validation.x.shape[1:]:
        raise ValueError("Training and validation sequence shapes differ")
    torch.set_num_threads(config['num_threads'])
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    target_mean = float(train.y.mean())
    target_scale = max(float(train.y.std()), 1.)
    model = RULLSTM(len(features), config['hidden_size'], config['num_layers'], config['dropout'],
                    target_mean=target_mean, target_scale=target_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(train.x, dtype=np.float32)),
                                     torch.from_numpy(train.y.astype(np.float32))),
                        batch_size=config['batch_size'], shuffle=True, generator=generator, num_workers=0)
    stopper = EarlyStopping(config['patience'], config['min_delta'])
    history = []
    start = time.perf_counter()
    predictor = LSTMPredictor(model, features, train.x.shape[1], batch_size=config['batch_size'])
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        squared_error = 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = (((prediction - y) / model.target_scale) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            squared_error += float(((prediction.detach().double() - y.double()) ** 2).sum())
        val_prediction = predictor.predict(validation.x, features)
        score = metrics(validation.y, val_prediction)
        val_mse = float(np.mean((val_prediction - validation.y) ** 2))
        history.append({'epoch': epoch, 'train_mse': squared_error / len(train.y),
                        'validation_mse': val_mse, 'validation_mae': score['mae'],
                        'validation_rmse': score['rmse']})
        print(f"Epoch {epoch}: train MSE={history[-1]['train_mse']:.3f}, validation RMSE={score['rmse']:.3f}", flush=True)
        if stopper.update(val_mse, model, epoch):
            break
    stopper.restore(model)
    model.eval()
    outcome = {'epochs_completed': len(history), 'best_epoch': stopper.best_epoch,
               'best_validation_mse': stopper.best_loss,
               'early_stopped': stopper.bad_epochs >= stopper.patience,
               'fit_seconds': time.perf_counter() - start,
               'target_mean': target_mean, 'target_scale': target_scale}
    return predictor, history, outcome


def train_lstm(settings):
    manifest, batches = load_prepared(settings)
    predictor, history, outcome = fit_lstm(batches['train'], batches['validation'], manifest['feature_names'],
                                            settings.models['lstm'], settings.random_seed)
    output = settings.artifact_dir / 'models' / settings.dataset / 'lstm'
    preprocessing = output / 'preprocessing'
    preprocessing.mkdir(parents=True, exist_ok=True)
    source = settings.artifact_dir / 'prepared' / settings.dataset
    shutil.copyfile(source / 'scaler.json', preprocessing / 'scaler.json')
    portable = deepcopy(manifest)
    portable['config'] = {k: v for k, v in portable['config'].items()
                          if k not in {'project_root', 'raw_data_dir', 'artifact_dir'}}
    (preprocessing / 'manifest.json').write_text(json.dumps(portable, indent=2) + '\n')
    predictor.preprocessing_sha256 = {name: digest(preprocessing / name) for name in ('manifest.json', 'scaler.json')}
    predictor.save(output)
    batch = batches['validation']
    prediction = predictor.predict(batch.x, manifest['feature_names'])
    restored = LSTMPredictor.load(output).predict(batch.x, manifest['feature_names'])
    if not np.array_equal(prediction, restored):
        raise ValueError('Saved checkpoint predictions differ')
    score = metrics(batch.y, prediction)
    if not np.isclose(score['rmse'] ** 2, outcome['best_validation_mse'], rtol=1e-12):
        raise ValueError('Best checkpoint was not restored correctly')
    result = batch.endpoints.copy()
    result['rul'], result['predicted_rul'] = batch.y, prediction
    result.to_csv(output / 'validation_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    report = {'model': 'lstm', 'architecture': predictor.model.architecture,
              'parameter_count': sum(p.numel() for p in predictor.model.parameters()),
              'model_config': settings.models['lstm'], 'seed': settings.random_seed,
              'device': 'cpu', 'mps_available': torch.backends.mps.is_available(),
              'deterministic_algorithms': True, 'optimizer': 'AdamW',
              'loss': 'MSE of training-standardized targets; history MSE in cycles squared',
              'selection': 'lowest development-validation MSE with configured min_delta and patience',
              'training_shape': list(batches['train'].x.shape), 'validation_shape': list(batch.x.shape),
              'training_engines': int(batches['train'].endpoints.unit_id.nunique()),
              'validation_engines': int(batch.endpoints.unit_id.nunique()),
              'source_sha256': manifest['source_sha256'],
              'prepared_sha256': {p.name: digest(p) for p in sorted(source.iterdir()) if p.is_file()},
              'checkpoint_sha256': digest(output / 'best.pt'),
              'versions': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pandas')},
              'python': platform.python_version(), **outcome, **score}
    (output / 'metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(train_lstm(load_config(args.config)), indent=2))


if __name__ == '__main__':
    main()
