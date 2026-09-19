# PlanWise ML Dashboard

PlanWise is a dynamic project-intelligence dashboard built around four tuned Random Forest models. It predicts task delay, classifies risk, estimates delay severity and effort overrun, monitors workload, and converts model output into actionable recommendations.

## What is included

- **Portfolio overview:** dynamic filters, health scoring, delay pressure, predicted risk, and an intervention queue.
- **Workload intelligence:** resource utilization, capacity states, delivery pressure, and allocation recommendations.
- **Prediction lab:** interactive what-if scoring for an individual task.
- **Model audit:** untouched test metrics, confusion matrices, feature importance, and tuning details.
- **Data workspace:** CSV validation, batch scoring/export, and in-app retraining.

## Model design

The training pipeline deliberately excludes IDs, target variables, actual effort, and actual duration from model inputs. It also groups all splits by `project_id`, preventing tasks from the same project from appearing in both training and test data.

| Model | Target | Evaluation |
|---|---|---|
| Random Forest classifier | `is_delayed` | ROC–AUC, precision, recall, F1, confusion matrix |
| Random Forest classifier | `risk_level` | Macro F1, balanced accuracy, confusion matrix |
| Random Forest regressor | `delay_days` on delayed tasks | MAE, RMSE, R² |
| Random Forest regressor | `effort_overrun_ratio` | MAE, RMSE, R² |

Hyperparameters are tuned with randomized parameter sampling and three-fold `GroupKFold`. Every candidate iteration reports its current and best score in both the terminal and the dashboard. The delay decision threshold is selected on a separate validation group and evaluated once on the untouched test group.

The supplied dataset's risk classes are perfectly separable from its risk fields. The dashboard calls this out explicitly because a perfect synthetic holdout result should not be interpreted as guaranteed real-world performance.

## Run locally

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --iterations 8
python -m streamlit run app.py
```

## Expected data

The bundled dataset is available at `data/planwise_ml_dataset.csv`. A replacement CSV can be uploaded from the dashboard. Batch scoring requires the 22 model input columns; retraining additionally requires `project_id`, `is_delayed`, `risk_level`, `delay_days`, and `effort_overrun_ratio`.

## Files

- `app.py` — interactive Streamlit dashboard
- `train.py` — command-line training entry point
- `src/planwise_ml.py` — training, validation, scoring, health, and recommendations
- `artifacts/` — trained model bundle, metrics, feature importance, and holdout predictions
- `tests/` — focused leakage, validation, health, and recommendation tests

## Responsible use

PlanWise is a decision-support system. Predictions should be reviewed by a project manager, monitored for data drift, and retrained when delivery processes or data definitions change.
