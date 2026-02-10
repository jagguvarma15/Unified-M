"""
Prior specification and warm-start utilities for Bayesian MMM backends.

Priors can come from three sources (in priority order):
  1. Experiment calibration factors  (dim_experiments table)
  2. Previous run's posteriors       (warm-start)
  3. Manual config                   (config.yaml per-channel)

Each Bayesian adapter (PyMC, Meridian, NumPyro) reads ChannelPrior
objects and maps them to its native prior API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class ChannelPrior:
    """Prior specification for a single channel."""

    channel: str

    # Coefficient (effect size)
    beta_mu: float = 0.0
    beta_sigma: float = 1.0

    # Adstock decay
    adstock_alpha_mu: float = 0.5
    adstock_alpha_sigma: float = 0.2

    # Saturation half-point
    saturation_K_mu: float = 5000.0
    saturation_K_sigma: float = 2000.0

    # Hill coefficient
    saturation_S_mu: float = 1.0
    saturation_S_sigma: float = 0.5

    # Calibration factor from experiments (None = no test data)
    calibration_factor: float | None = None
    calibration_confidence: float | None = None

    # Source of the prior
    source: str = "default"  # default | config | warm_start | experiment

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "beta_mu": self.beta_mu,
            "beta_sigma": self.beta_sigma,
            "adstock_alpha_mu": self.adstock_alpha_mu,
            "adstock_alpha_sigma": self.adstock_alpha_sigma,
            "saturation_K_mu": self.saturation_K_mu,
            "saturation_K_sigma": self.saturation_K_sigma,
            "saturation_S_mu": self.saturation_S_mu,
            "saturation_S_sigma": self.saturation_S_sigma,
            "calibration_factor": self.calibration_factor,
            "source": self.source,
        }


@dataclass
class PriorSet:
    """Collection of channel priors for a model run."""

    channel_priors: dict[str, ChannelPrior] = field(default_factory=dict)

    def get(self, channel: str) -> ChannelPrior:
        """Return prior for channel, or a default prior."""
        return self.channel_priors.get(channel, ChannelPrior(channel=channel))

    def to_dict(self) -> dict[str, Any]:
        return {ch: p.to_dict() for ch, p in self.channel_priors.items()}

    def save(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "PriorSet":
        with open(path) as f:
            data = json.load(f)
        priors = {}
        for ch, d in data.items():
            priors[ch] = ChannelPrior(**d)
        return cls(channel_priors=priors)


def build_default_priors(channels: list[str]) -> PriorSet:
    """Create default (vague) priors for a set of channels."""
    priors = {}
    for ch in channels:
        priors[ch] = ChannelPrior(channel=ch, source="default")
    return PriorSet(channel_priors=priors)


def warm_start_from_run(
    run_dir: Path | str,
    channels: list[str],
    shrink_factor: float = 1.5,
) -> PriorSet:
    """
    Create informative priors from a previous run's parameters.

    The posterior means become the new prior means, and the posterior
    uncertainty is widened by ``shrink_factor`` to allow the new run
    to explore.

    Args:
        run_dir:        Path to a previous run directory.
        channels:       Channels to extract priors for.
        shrink_factor:  Multiply posterior SD by this to widen priors.

    Returns:
        PriorSet populated from the previous run.
    """
    run_dir = Path(run_dir)
    params_path = run_dir / "parameters.json"

    if not params_path.exists():
        logger.warning(f"No parameters.json in {run_dir}, falling back to defaults")
        return build_default_priors(channels)

    with open(params_path) as f:
        params = json.load(f)

    coefficients = params.get("coefficients", {})
    adstock_params = params.get("adstock_params", {})
    saturation_params = params.get("saturation_params", {})

    priors = {}
    for ch in channels:
        coef = coefficients.get(ch, 0.0)
        adstock = adstock_params.get(ch, {})
        saturation = saturation_params.get(ch, {})

        priors[ch] = ChannelPrior(
            channel=ch,
            beta_mu=coef,
            beta_sigma=abs(coef) * 0.3 * shrink_factor + 0.01,
            adstock_alpha_mu=adstock.get("alpha", 0.5),
            adstock_alpha_sigma=0.15 * shrink_factor,
            saturation_K_mu=saturation.get("K", 5000.0),
            saturation_K_sigma=saturation.get("K", 5000.0) * 0.3 * shrink_factor,
            saturation_S_mu=saturation.get("S", 1.0),
            saturation_S_sigma=0.3 * shrink_factor,
            source="warm_start",
        )

    logger.info(f"Built warm-start priors from {run_dir} for {len(priors)} channels")
    return PriorSet(channel_priors=priors)


def apply_calibration_factors(
    priors: PriorSet,
    calibration_factors: dict[str, float],
    calibration_confidence: dict[str, float] | None = None,
) -> PriorSet:
    """
    Adjust priors using experiment calibration factors.

    If a channel has calibration_factor = 1.3, it means the experiment
    measured 30% more lift than the MMM predicted.  We shift the prior
    mean accordingly.

    Args:
        priors:                 Base prior set.
        calibration_factors:    Channel -> test_lift / mmm_lift ratio.
        calibration_confidence: Channel -> confidence in the test (0-1).

    Returns:
        Updated PriorSet.
    """
    conf = calibration_confidence or {}

    for ch, factor in calibration_factors.items():
        if ch in priors.channel_priors:
            p = priors.channel_priors[ch]
            p.beta_mu *= factor
            p.calibration_factor = factor
            p.calibration_confidence = conf.get(ch, 0.8)
            p.source = "experiment"

            # Tighten sigma when we have high-confidence test data
            confidence = conf.get(ch, 0.8)
            p.beta_sigma *= (1.0 - 0.3 * confidence)  # tighter with more confidence

            logger.info(
                f"Applied calibration factor {factor:.2f} to {ch} "
                f"(new beta_mu={p.beta_mu:.4f})"
            )

    return priors
