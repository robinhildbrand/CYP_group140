"""Models selectable with model.type. A model is an nn.Module taking
(model_cfg, readout_cfg, n_targets, data_dir, device) with fit_preprocessing(train),
loader(frame, targets, batch_size, shuffle), param_groups() and
forward(batch) -> [batch, n_targets] predictions. Readout heads are in models/readouts.py.
The LightGBM baseline (models/gbm.py, model.type=lightgbm) is handled separately in finetune.py.

An entry is imported when it is selected, not when this module loads, so a model's own
third-party dependency -- atom-jepa for the Atom-JEPA family, transformers for ChemBERTa --
is needed only by the runs that actually use it, and a missing one names the model instead of
breaking every run in the project. finetune.py only ever calls MODELS[model.type](...)."""
import importlib


def _model(module, name):
    def build(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)
    return build


MODELS = {'atom_jepa': _model('models.atom_jepa', 'AtomJEPAModel'),
          'features': _model('models.baselines', 'FeatureModel'),
          'atom_jepa_cached': _model('models.atom_jepa_cached', 'CachedAtomJEPAModel'),
          'atom_jepa_ft': _model('models.atom_jepa_ft', 'FineTunedAtomJEPAModel'),
          'atom_jepa_tdi': _model('models.atom_jepa_tdi', 'TDIAtomJEPAModel'),
          'chemberta': _model('models.chemberta', 'ChemBERTa')}
