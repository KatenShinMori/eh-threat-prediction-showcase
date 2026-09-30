# EW-Resilient Multi-Sensor Threat Tracking & Trajectory Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Aerodynamics](https://img.shields.io/badge/Aerodynamics-JSBSim%206--DoF-0A84FF.svg)](https://github.com/JSBSim-Team/jsbsim)
[![ONNX](https://img.shields.io/badge/Inference-ONNX%20%7C%20TensorRT-005CED.svg)](https://onnx.ai/)

Target tracking algorithms usually break down when exposed to active Electronic Warfare (EW)—such as radar noise jamming, range-gate pull-off (RGPO), or intermittent packet drops. Classical filters (EKF) diverge, while generic deep learning baselines (LSTM) often output trajectories that violate basic aerodynamics (e.g., predicting an aircraft pulling 20G+ maneuvers).

This repository showcases a hybrid deep learning pipeline designed to solve both problems:
1. **Cross-Attention Transformer:** Fuses asynchronous, multi-rate sensor inputs (Radar, EO/IR, ESM) and uses reliability gating to automatically down-weight jammed sensors.
2. **Physics-Informed Neural Network (PINN):** Continuously forecasts future trajectory ($t \in [0, 5\text{s}]$) while penalizing load factors exceeding $9\text{G}$ and enforcing realistic flight envelopes.
3. **JSBSim 6-DoF Simulation:** Validates performance against an F-16 flight dynamics model under tactical maneuvers.

---

## Quick Start / Architecture Verification

A self-contained reference implementation of the core neural architecture is provided in `demo_pipeline.py`. It requires only PyTorch to run:

```bash
python demo_pipeline.py
```

This script verifies:
- Continuous **Time2Vec** embeddings on asynchronous timestamps.
- **Reliability gating** de-weighting corrupted sensor channels under EW noise.
- Continuous-time **$C^1$ boundary pinning** ($p(0) = p_0, \dot{p}(0) = v_0$).
- Exact autograd derivatives ($\mathbf{v} = \dot{\mathbf{p}}$, $\mathbf{a} = \ddot{\mathbf{p}}$) enforcing $|N_z| \le 9.0\text{ G}$.

---

## Architecture

```mermaid
flowchart TD
    subgraph Inputs ["Asynchronous Sensor Streams"]
        direction TB
        R["Radar (~10 Hz)<br/>Range, Az/El, Doppler"]
        EO["EO/IR (~30 Hz)<br/>Bearing, Elevation"]
        ESM["RF / ESM (~5 Hz)<br/>AOA, Signal Strength"]
        EW["EW Environment<br/>Noise Jamming / Dropped Packets"] -.->|Corrupts| R
    end

    subgraph Fusion ["1. Cross-Attention Transformer"]
        direction TB
        Tok["Continuous Tokenizer<br/>(Time2Vec + Sensor Embeddings)"]
        Gate["Cross-Attention + Reliability Gate<br/>(Penalizes high-variance / jammed inputs)"]
        Tok --> Gate
    end

    subgraph Predictor ["2. Continuous-Time PINN Decoder"]
        direction TB
        MLP["Latent State -> Trajectory Head"]
        Phys["Autograd Differential Constraints<br/>• Load factor |Nz| <= 9G<br/>• Energy & velocity bounds"]
        MLP --> Phys
    end

    Inputs --> Tok
    Gate --> MLP
    Phys --> Out["Output: 5-Second Forecast (Sub-10ms Latency)"]

    style EW stroke:#e74c3c,stroke-width:2px,stroke-dasharray: 5 5
    style Fusion fill:#161b22,stroke:#30363d
    style Predictor fill:#161b22,stroke:#30363d
```

---

## Why This Works

* **Asynchronous clocks without fixed resampling:** Radar (10Hz), EO/IR (30Hz), and ESM (5Hz) run at different rates. Using continuous Time2Vec representations lets the network process sensor packets whenever they arrive instead of forcing brittle interpolation.
* **Soft sensor isolation:** When radar jamming turns on, the reliability gate drops the attention weight on radar tokens and relies primarily on passive EO/IR and RF bearings.
* **Hard kinematic boundary pinning:** By formulating the decoder as:
  $$p(t) = p_0 + v_0 t + t^2 \cdot \Delta p_\theta(t)$$
  The trajectory at $t=0$ identically equals the estimated position $p_0$, and its first derivative $\dot{p}(0)$ identically equals $v_0$. This guarantees zero jump discontinuities at the handover boundary.
* **Physics limits via autograd:** Accelerations and velocities are computed analytically inside the network graph ($v = \dot{p}$, $a = \ddot{p}$). If the network attempts to bend a trajectory sharper than the aircraft's aerodynamic capability ($|N_z| > 9.0\text{ G}$), the loss heavily penalizes it during training.

---

## Benchmark Results

Tested on a standardized benchmark dataset of 100 tactical flight episodes (100-step observation history, 50-step forecast horizon).

### Tracking Error (RMSE in meters)

| Model | Condition | 1.0s Horizon | 3.0s Horizon | 5.0s Horizon | Phys. Violations | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PINN-Transformer** | **All** | **29.7 m** | **46.6 m** | **51.7 m** | **0.00%** | **9.08 ms** |
| **PINN-Transformer** | **Clean** | **27.0 m** | **44.1 m** | **48.3 m** | **0.00%** | **9.08 ms** |
| **PINN-Transformer** | **Jammed (EW)** | **34.3 m** | **51.2 m** | **57.7 m** | **0.00%** | **9.08 ms** |
| LSTM Baseline | Clean | 93.5 m | 69.6 m | 82.0 m | 0.00% | 3.52 ms |
| LSTM Baseline | Jammed (EW) | 280.7 m | 269.9 m | 268.2 m | 0.00% | 3.52 ms |
| Singer 9-State EKF | Clean | 45.5 m | 141.6 m | 329.1 m | 0.00% | 2.95 ms |
| Singer 9-State EKF | Jammed (EW) | 51.9 m | 147.7 m | 336.4 m | 0.00% | 2.95 ms |

> **Note on LSTM Baseline Dynamics:**  
> The unconstrained LSTM baseline exhibits higher error at 1.0s (93.5m / 280.7m) than at 3.0s and 5.0s. This is a known failure mode of unanchored sequence-to-vector regression: because the LSTM predicts absolute coordinates from hidden states without an initial kinematic anchor ($p_0 + v_0 t$), corrupted sensor inputs under EW jamming induce an immediate **offset shock at $t=1\text{s}$**. As the maneuvering aircraft travels forward into that spatial envelope over 3s and 5s, the Euclidean distance temporarily plateaus. This failure mode directly motivated our PINN formulation, where the boundary condition $p(t) = p_0 + v_0 t + t^2 \Delta p_\theta(t)$ enforces monotonic, physics-consistent error growth.

**Key Takeaways:**
1. **Under jamming**, EKF diverges quickly ($>330\text{ m}$ at 5s) because corrupted measurements contaminate its state covariance.
2. The LSTM baseline degrades to $268\text{ m}$ under jamming and suffers from initial handover discontinuities.
3. The **PINN-Transformer holds error to $57.7\text{ m}$** at 5.0s under jamming—a **~5x improvement** over both baselines.
4. Total forward-pass latency is **9.08 ms**, making it viable for 100Hz real-time avionics loops.

---

## Visualizations

### Tracking Error Progression
![RMSE Comparison](assets/rmse_comparison.png)

### Clean vs. Jammed Sensor Scenarios
![Clean vs Jammed](assets/rmse_clean_vs_jammed.png)

### Sensor Attention Weights During Active Jamming
The cross-attention layer automatically de-prioritizes jammed sensor channels:
![Attention Weights](assets/attention_weights.png)

---

## Aerodynamic Maneuver Validation (JSBSim)

Flight truth profiles generated with JSBSim 6-DoF F-16 dynamics under standard military maneuvers:

| Coordinated Turn ($30^\circ$ Bank) | Climb / Dive Profile |
| :---: | :---: |
| ![Turn Maneuver](assets/jsbsim_turn_maneuver.png) | ![Climb Dive Maneuver](assets/jsbsim_climb_dive_maneuver.png) |

| S-Turn Reversal | Trimmed Level Flight |
| :---: | :---: |
| ![Reversal Maneuver](assets/jsbsim_reversal_maneuver.png) | ![Level Flight](assets/jsbsim_level_flight.png) |

---

## Tech Stack

* **Frameworks:** PyTorch, NumPy, SciPy
* **Simulation:** JSBSim Flight Dynamics Engine (F-16 model, WGS-84 $\to$ ENU coordinates)
* **Target Export:** ONNX Runtime / TensorRT

---

## Roadmap

- [x] Cross-attention fusion architecture + Time2Vec continuous tokenization.
- [x] Continuous-time PINN decoder with differential autograd physics loss.
- [x] JSBSim 6-DoF aerodynamic maneuver simulator and dataset generation.
- [ ] Direct closed-loop training against the multi-threaded JSBSim flight pool.
- [ ] Embedded hardware profiling (NVIDIA Jetson Orin via TensorRT FP16).
- [ ] Multi-target track correlation and swarm scenarios.

---

<sub>*Note: This repository is a technical showcase containing architecture documentation, benchmark results, and an executable reference demo in `demo_pipeline.py`. Production mission simulation engines and proprietary training checkpoints are maintained internally.*</sub>