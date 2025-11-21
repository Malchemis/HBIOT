# Lazy imports - only load when actually used
# This dramatically speeds up startup time by avoiding heavy dependencies
# like scipy, mne, torchmetrics until they're actually needed

def __getattr__(name):
    """Lazy import modules on first access."""
    # Data module
    if name in ('MEGOnTheFlyDataModule', 'PredictionDataModule', 'create_datamodule'):
        from pipeline.data import MEGOnTheFlyDataModule, PredictionDataModule, create_datamodule
        globals()['MEGOnTheFlyDataModule'] = MEGOnTheFlyDataModule
        globals()['PredictionDataModule'] = PredictionDataModule
        globals()['create_datamodule'] = create_datamodule
        return globals()[name]

    # Models
    elif name in ('BIOTHierarchicalClassifier', 'BIOTClassifier', 'SFCN', 'FAMEDWrapper'):
        from pipeline.models import BIOTHierarchicalClassifier, BIOTClassifier, SFCN, FAMEDWrapper
        globals()['BIOTHierarchicalClassifier'] = BIOTHierarchicalClassifier
        globals()['BIOTClassifier'] = BIOTClassifier
        globals()['SFCN'] = SFCN
        globals()['FAMEDWrapper'] = FAMEDWrapper
        return globals()[name]

    # Evaluation
    elif name == 'MetricsAggregator':
        from pipeline.eval import MetricsAggregator
        globals()['MetricsAggregator'] = MetricsAggregator
        return MetricsAggregator

    # Optimization
    elif name in ('create_loss', 'create_optimizer', 'create_scheduler'):
        from pipeline.optim import create_loss, create_optimizer, create_scheduler
        globals()['create_loss'] = create_loss
        globals()['create_optimizer'] = create_optimizer
        globals()['create_scheduler'] = create_scheduler
        return globals()[name]

    # Training
    elif name in ('create_callbacks', 'MEGSpikeDetector'):
        from pipeline.training import create_callbacks, MEGSpikeDetector
        globals()['create_callbacks'] = create_callbacks
        globals()['MEGSpikeDetector'] = MEGSpikeDetector
        return globals()[name]

    # Utils
    elif name == 'load_config':
        from pipeline.utils import load_config
        globals()['load_config'] = load_config
        return load_config

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")