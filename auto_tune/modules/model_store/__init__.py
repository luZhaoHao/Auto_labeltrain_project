"""F1.1-A 受控权重库：列举、上传、身份解析与提交前复核。

权重是不透明字节：本包不反序列化、不联网、不下载，也不扫描训练产物。
"""

from .service import (
    MODEL_CHANGED,
    MODEL_NAME_CONFLICT,
    MODEL_NOT_FOUND,
    MODEL_PATH_FORBIDDEN,
    MODEL_PATH_UNSAFE,
    MODEL_STORE_UNAVAILABLE,
    MODEL_UPLOAD_EMPTY,
    MODEL_UPLOAD_INVALID_NAME,
    MODEL_UPLOAD_INVALID_TYPE,
    MODEL_UPLOAD_TOO_LARGE,
    ModelRecord,
    ModelStore,
    ModelStoreError,
)

__all__ = [
    "ModelRecord",
    "ModelStore",
    "ModelStoreError",
    "MODEL_UPLOAD_INVALID_NAME",
    "MODEL_UPLOAD_INVALID_TYPE",
    "MODEL_UPLOAD_EMPTY",
    "MODEL_UPLOAD_TOO_LARGE",
    "MODEL_NAME_CONFLICT",
    "MODEL_STORE_UNAVAILABLE",
    "MODEL_NOT_FOUND",
    "MODEL_CHANGED",
    "MODEL_PATH_UNSAFE",
    "MODEL_PATH_FORBIDDEN",
]
