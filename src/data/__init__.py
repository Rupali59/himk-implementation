# Data sources: filesystem or KTH-like DB (e.g. MongoDB)

from .loader import get_data_loader, list_splits

__all__ = ["get_data_loader", "list_splits"]
