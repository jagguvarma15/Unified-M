"""
Unified-M: Unified Marketing Measurement Framework.

Produces a single incrementality truth by reconciling MMM + attribution
signals + lift/geo tests into one consistent set of channel contributions
with uncertainty.

Quickstart::

    from pipeline import Pipeline
    pipe = Pipeline()
    pipe.connect(media_spend="data/media.csv", outcomes="data/outcomes.csv")
    results = pipe.run(model="builtin")
"""

__version__ = "0.2.0"
