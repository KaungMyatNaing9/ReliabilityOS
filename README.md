# ReliabilityOS

### ABB Hackathon — Theme 1: Agentic Predictive Maintenance Studio

**ReliabilityOS** is an agentic AutoML and MLOps demo that helps industrial teams build, validate, monitor, and improve predictive-maintenance models with less manual machine-learning work.

Industrial facilities produce high volumes of data from motors, drives, pumps, and other critical equipment. Turning that telemetry into a dependable predictive-maintenance system normally requires specialists to inspect data, engineer features, test models, validate reliability, deploy a champion, then rebuild the pipeline whenever operating conditions change.

ReliabilityOS turns that process into an evidence-backed reliability loop while keeping engineers in control of critical promotion decisions.

> Machine data → Profile → Train → Validate → Deploy → Monitor → Detect drift → Improve → Compare → Human approve

## The problem

Predictive-maintenance projects often stall because a high accuracy score alone is not enough. In an industrial setting, a useful model must catch failures, avoid overwhelming technicians with false alarms, respond quickly, remain calibrated, and continue working when sensor distributions change.

When machines operate under new loads, temperatures, materials, or maintenance regimes, the original model can silently degrade. Teams need a repeatable way to detect that shift, investigate it, test an alternative, and govern the rollout.

## What ReliabilityOS does

The demo provides one autonomous workflow that:

1. Loads bundled predictive-maintenance telemetry or a CSV upload.
2. Profiles schema, missingness, timestamps, asset IDs, labels, class balance, and the likely ML task.
3. Builds leakage-safe, time-aware features and chronological train/validation/test splits.
4. Trains Logistic Regression, Random Forest, and XGBoost when available, with a sklearn boosting fallback.
5. Selects a champion using failure recall, precision, F1, false-alarm rate, ROC-AUC, Brier calibration score, and inference latency—not accuracy alone.
6. Runs an independent Reliability Critic for leakage, temporal-split validity, imbalance, missing data, calibration, recall, false alarms, and latency.
7. Deploys the accepted champion to a clearly labelled simulated production environment.
8. Simulates operating-condition drift, measures PSI sensor drift, and evaluates labelled production degradation.
9. Produces a drift-informed challenger experiment, retrains it on the protected temporal split, compares it with the champion, and re-runs the Critic.
10. Requires explicit human approval before promotion and records the event in local experiment history and the knowledge graph.

## Why it matters for industry

- **Less unplanned downtime:** failure recall focuses the model on finding maintenance risks before they become outages.
- **Less alert fatigue:** false-alarm rate is a first-class selection and governance metric.
- **Faster adoption:** the workflow removes repetitive profiling, preprocessing, evaluation, and experiment-comparison work.
- **Traceable decisions:** Critic evidence, candidate metrics, drift reports, and human approvals make model changes easier to audit.
- **Safer continuous improvement:** models do not automatically replace production systems; every challenger is validated and human-gated.
- **Reusable reliability knowledge:** model, feature, drift, asset, and promotion relationships are captured in a lightweight knowledge graph.

## How the self-improvement loop works

ReliabilityOS does **not** claim to continuously fine-tune a foundation model or rewrite its own code. It improves the predictive-maintenance ML pipeline through controlled experimentation:

1. The monitor compares production sensor distributions against the training baseline using Population Stability Index (PSI).
2. A deterministic improvement agent identifies the highest-drift sensors and the champion’s metric weaknesses.
3. It creates a bounded challenger configuration with changed model parameters and/or feature strategy.
4. The challenger trains on the same leakage-safe temporal split and is evaluated with the same industrial metrics.
5. The Reliability Critic independently validates the challenger.
6. A reliability engineer approves or rejects promotion; approved outcomes become experiment knowledge for the next investigation.

## Product experience

The Streamlit app is intentionally a simple, one-screen experience:

- **Run autonomous demo** executes the end-to-end happy path with bundled deterministic factory telemetry.
- **Findings** summarize the selected champion, Critic disposition, production drift, observed degradation, and challenger recommendation.
- **Model and monitoring plots** show candidate quality, native feature importance, champion-versus-drift-window performance, and sensor PSI scores.
- **Human approval** is the only manual gate required to promote the challenger.

The bundled factory telemetry, drift event, and labelled production impact are simulations. The ML, data profiling, reliability checks, drift detection, challenger training, comparison, and approval workflow are real local Python logic.

## Technical approach

| Area | Implementation |
| --- | --- |
| UI | Streamlit single-screen industrial dashboard |
| Data intelligence | pandas profiling, task inference, temporal splitting, leakage-safe rolling features |
| AutoML | scikit-learn Logistic Regression and Random Forest; XGBoost when installed; Gradient Boosting fallback |
| Reliability evaluation | recall, precision, F1, false-alarm rate, ROC-AUC, Brier score, inference latency |
| Model governance | deterministic Reliability Critic with evidence-backed PASS / WARN / FAIL decisions |
| Monitoring | deterministic drift simulation and Population Stability Index (PSI) |
| Explainability | native tree-feature importance or absolute linear coefficients |
| Knowledge | NetworkX graph plus local JSON promotion-history audit trail |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, then click **Run autonomous demo**. The bundled dataset is available at `data/sample_predictive_maintenance.csv` and is regenerated deterministically if missing.

## Validation

```bash
python -m pytest -q
python -m compileall .
```

The automated test suite covers the data pipeline, model metrics, critic checks, drift and challenger loop, model evidence, local history, knowledge graph, and the full demo happy path.

## Future direction

The current demo is local and deterministic by design. A production version could connect governed asset telemetry, support shadow-deployment windows, integrate maintenance work-order feedback, add role-based approvals, and reuse validated strategies across equipment fleets.
