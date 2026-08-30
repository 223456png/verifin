"""ui 子包：终端用户可见的中文诊断消息模板（不依赖 LangGraph）。"""

from verifin.ui.messages import (
    friendly_message,
    metric_label,
    preference_confirmation,
    source_label,
)

__all__ = ["friendly_message", "metric_label", "preference_confirmation", "source_label"]