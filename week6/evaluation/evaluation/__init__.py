"""Week6 评估包"""

from .metrics import (
    full_evaluation,
    compute_predict_metrics,
    compute_anomaly_rationality,
    aggregate_events,
    compute_classification_metrics,
    hour_of_t,
    period_id_of_t,
    is_core_cell,
)
from .profile_runner import (
    profile_pipeline,
    profile_batch,
    profile_api_endpoint,
    get_ram_mb,
    get_gpu_mb,
)
