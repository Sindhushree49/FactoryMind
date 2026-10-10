"""Unseen-data smoke tests for FactoryMind inference (not part of the app)."""

from __future__ import annotations

import numpy as np

from app.services import ml_service

fm = ml_service.failure_model()
am = ml_service.anomaly_model()

cases = [
    {"machine_type": "L", "air_temperature_k": 300.5, "process_temperature_k": 310.6,
     "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 10},
    {"machine_type": "M", "air_temperature_k": 320.0, "process_temperature_k": 345.0,
     "rotational_speed_rpm": 2400, "torque_nm": 55.0, "tool_wear_min": 200},
    {"machine_type": "H", "air_temperature_k": 295.0, "process_temperature_k": 305.0,
     "rotational_speed_rpm": 1200, "torque_nm": 25.0, "tool_wear_min": 5},
]

for i, c in enumerate(cases, 1):
    r = fm.predict(c)
    print(f"failure #{i}: p={r['failure_probability']:.4f} risk={r['risk_category']} "
          f"mode={r['predicted_failure_type']} shap_features={len(r['features'])}")

for i, c in enumerate(cases, 1):
    r = am.predict(c)
    print(f"anomaly #{i}: score={r['anomaly_score']:.4f} threshold={r['threshold']} "
          f"is_anomaly={r['is_anomaly']} severity={r['severity']}")

extreme = {"machine_type": "L", "air_temperature_k": 250.0, "process_temperature_k": 355.0,
           "rotational_speed_rpm": 29000, "torque_nm": 59.0, "tool_wear_min": 255}
r = am.predict(extreme)
print(f"extreme anomaly: score={r['anomaly_score']:.4f} is_anomaly={r['is_anomaly']} "
      f"severity={r['severity']}")

# RUL families
from app.ml.rul_prediction.inference import RULInference

for family in ("xgboost", "lstm"):
    inf = RULInference(family=family)
    rng = np.random.default_rng(7)
    synthetic = rng.normal(loc=0.0, scale=1.0, size=(30, 17)).astype(np.float32)
    res = inf.predict_window(synthetic, subset="FD001", engine_unit="unseen", cycle=99)
    assert np.isfinite(res["predicted_rul"])
    print(f"rul {family}: rul={res['predicted_rul']:.2f} finite=True")

print("ALL INFERENCE SMOKE TESTS PASSED")
