"""模型注册 — 通过 name 索引到对应的 base trainer 类"""
from metrics import BaseTrainer

# 延迟 import（ARIMA/Prophet 用 statsmodels/prophet, 较慢）
def get_trainer(name: str) -> BaseTrainer:
    name = name.lower()
    if name == "arima":
        from models.arima_runner import ARIMABaseTrainer
        return ARIMABaseTrainer()
    elif name == "prophet":
        from models.prophet_runner import ProphetBaseTrainer
        return ProphetBaseTrainer()
    elif name == "lstm":
        from models.lstm_model import LSTMBaseTrainer
        return LSTMBaseTrainer()
    elif name == "gru":
        from models.gru_model import GRUBaseTrainer
        return GRUBaseTrainer()
    elif name == "gcn":
        from models.gcn_model import GCNBaseTrainer
        return GCNBaseTrainer()
    elif name == "gru_stgcn_residual":
        from models.gru_stgcn_residual import GRU_STGCN_ResidualTrainer
        return GRU_STGCN_ResidualTrainer()
    elif name == "gat":
        from models.gat_model import GATBaseTrainer
        return GATBaseTrainer()
    else:
        raise ValueError(f"unknown model: {name}")


def list_models():
    return ["arima", "prophet", "lstm", "gru", "gcn", "gat", "gru_stgcn_residual"]
