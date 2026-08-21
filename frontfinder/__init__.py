"""frontfinder: batch inference + zarr serving pipeline for the ESPR "Fronts" product.

Runs Keras front-detection models against streamed ECMWF IFS open-data
global 0.25 deg fields and writes results to a zarr multiscale pyramid for
the fronts.espr.ai maplibre/topozarr viewer.

Built for two models (best_loss, model_1702), but model_1702 is currently
disabled -- its trained pressure levels [1000, 950, 900, 850] include
950/900hPa, which IFS open-data's 0.25deg feed doesn't publish. See
frontfinder/scheduler/cli.py's module docstring for the full explanation
and how to re-enable it once that gap has a chosen resolution.
"""

__version__ = "0.1.0"
