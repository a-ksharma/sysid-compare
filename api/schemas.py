"""Pydantic v2 schemas for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Standard error response detail."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")


class ErrorResponse(BaseModel):
    """Top-level error response envelope."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check status and loaded system information."""

    status: str = Field(default="ok", description="Service health status")
    version: str = Field(default="0.1.0", description="API version")
    loaded_plants: list[str] = Field(
        ..., description="List of loaded physical plant identifiers"
    )
    loaded_models: dict[str, list[str]] = Field(
        ..., description="Available identification models per plant"
    )


class SystemSummary(BaseModel):
    """Summary of a physical plant system."""

    id: str = Field(..., description="Unique plant identifier ('rc', 'motor', 'msd')")
    name: str = Field(..., description="Descriptive system name")
    order: int = Field(..., description="System polynomial order")
    nominal_parameters: dict[str, float] = Field(
        ..., description="Nominal physical parameters with units"
    )
    u_name: str = Field(..., description="Input variable name")
    u_unit: str = Field(..., description="Input physical unit")
    y_name: str = Field(..., description="Output variable name")
    y_unit: str = Field(..., description="Output physical unit")
    u_max: float = Field(..., description="Nominal maximum input amplitude")
    training_band_rad_s: list[float] = Field(
        ..., description="Excitation frequency band [omega_min, omega_max] in rad/s"
    )


class PoleComplex(BaseModel):
    """Continuous or discrete pole in real/imag representation."""

    real: float = Field(..., description="Real component")
    imag: float = Field(..., description="Imaginary component")


class SystemDetail(SystemSummary):
    """Detailed plant parameters including true and identified values."""

    true_poles: list[PoleComplex] = Field(..., description="True continuous-time poles")
    identified: dict[str, Any] = Field(
        ..., description="Identified parameters and poles per model"
    )


class ModelCardResponse(BaseModel):
    """Model metadata cards and training provenance across plants."""

    cards: dict[str, Any] = Field(
        ..., description="Mapping of plant_id to dictionary of model cards"
    )


class InputSpecification(BaseModel):
    """Specification of input signal to drive simulation."""

    kind: Literal["step", "sine", "square", "chirp", "multisine", "custom"] = Field(
        ..., description="Waveform family"
    )
    amplitude: float = Field(
        default=1.0, description="Input amplitude in physical units"
    )
    frequency_rad_s: float | None = Field(
        default=None, description="Signal frequency for sine/square/chirp in rad/s"
    )
    samples: list[float] | None = Field(
        default=None, description="Custom input sequence samples at Ts"
    )


class SimulateRequest(BaseModel):
    """Request payload for plant and model simulation."""

    system_id: str = Field(
        ..., description="Plant identifier to simulate ('rc', 'motor', 'msd')"
    )
    models: list[str] = Field(
        default=["arx", "prony", "neural_ode"],
        description="List of models to evaluate against true plant",
    )
    input: InputSpecification = Field(
        ..., description="Input excitation signal specification"
    )
    duration_s: float = Field(
        default=15.0, description="Simulation duration in seconds (max 3000 samples)"
    )
    noise_snr_db: float | None = Field(
        default=None,
        description="Optional output Gaussian measurement noise SNR in dB",
    )
    seed: int = Field(
        default=1, description="Pseudorandom seed for measurement noise generation"
    )


class ModelPrediction(BaseModel):
    """Predicted trajectory and performance metrics for one model."""

    y: list[float] = Field(..., description="Simulated output trajectory")
    fit_percent: float = Field(..., description="Normalized goodness-of-fit percent")
    rmse: float = Field(..., description="Root-mean-squared error in physical units")


class SimulateResponse(BaseModel):
    """Simulation trajectory results, metrics, and out-of-range warnings."""

    ts: float = Field(..., description="Sampling interval Ts in seconds")
    t: list[float] = Field(..., description="Time grid points in seconds")
    u: list[float] = Field(..., description="Input signal trajectory")
    y_true: list[float] = Field(..., description="Ground truth clean plant output")
    y_measured: list[float] | None = Field(
        default=None,
        description="Noisy measured output when noise_snr_db is specified",
    )
    predictions: dict[str, ModelPrediction] = Field(
        ..., description="Simulation predictions per requested model"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages if input leaves nominal training range",
    )


class BodeCurve(BaseModel):
    """Magnitude and phase frequency response curves."""

    magnitude_db: list[float] = Field(..., description="Magnitude in decibels")
    phase_deg: list[float] = Field(..., description="Phase in degrees")


class FrequencyResponseData(BaseModel):
    """Frequency response Bode data for true plant and identified models."""

    omega: list[float] = Field(..., description="Frequency points in rad/s")
    true_system: BodeCurve = Field(..., description="True plant frequency response")
    models: dict[str, BodeCurve] = Field(
        ..., description="Bode curves per requested model"
    )


class ImpulseCurve(BaseModel):
    """Impulse response trajectory."""

    h: list[float] = Field(..., description="Impulse response values")


class ImpulseResponseData(BaseModel):
    """Impulse response data for true plant and identified models."""

    t: list[float] = Field(..., description="Time points in seconds")
    true_system: list[float] = Field(..., description="True plant impulse response")
    models: dict[str, list[float]] = Field(
        ..., description="Impulse response values per requested model"
    )
