import os
import time
from typing import Any, Optional, cast

from transformers import AutoModelForCausalLM, AutoTokenizer


def _default_proxy_cfg_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "proxy",
        "svr.ini",
    )


def load_model_and_tokenizer(
    model_name: str, dtype: Optional[Any] = None
) -> tuple[Any, Any]:
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype),
    )
    model.eval()
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name))
    return model, tokenizer


def start_backend(
    backend_name: str,
    block_size: int = 1024,
    cfg_path: Optional[str] = None,
    enable_cache: bool = False,
    loop: Optional[Any] = None,
) -> Any:
    from morphling.api import AutoBackend

    if backend_name == "proxy":
        backend = cast(Any, AutoBackend.from_name(backend_name))
        if enable_cache and hasattr(backend, "set_cache_enabled"):
            backend.set_cache_enabled(True)  # type: ignore[attr-defined]
        config_path = cfg_path or _default_proxy_cfg_path()
        backend.initialize(config_path)  # type: ignore[attr-defined]
        backend.start()  # type: ignore[attr-defined]
        return backend

    raise ValueError(f"Unknown backend: {backend_name}")


def _get_connection_count_safe(backend: Any) -> int:
    if hasattr(backend, "get_connection_count"):
        return backend.get_connection_count()  # type: ignore[attr-defined]
    return 0


def wait_for_connections(
    backend: Any, min_devices: int, timeout: int = 120
) -> int:
    if not hasattr(backend, "get_connection_count"):
        print("Backend does not support connection counting")
        return 0

    print(
        f"Waiting for at least {min_devices} device(s) to connect (timeout: {timeout}s)..."
    )
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            connection_count = _get_connection_count_safe(backend)
            elapsed = int(time.time() - start_time)
            print(
                f"[{elapsed}s] Connected devices: {connection_count}/{min_devices}"
            )

            if connection_count >= min_devices:
                print(f"✓ {connection_count} device(s) connected!")
                return connection_count

            time.sleep(2)
        except Exception as e:
            print(f"Error checking connection count: {e}")
            time.sleep(2)

    final_count = _get_connection_count_safe(backend)
    print(
        f"Timeout after {timeout}s. Connected: {final_count}/{min_devices} devices"
    )
    return final_count


def prepare_inputs(tokenizer: Any, batch_size: int, seq_length: int) -> Any:
    input_text = [
        "".join("Hello, my dog is cute. He is a good ") * 128
    ] * batch_size
    return tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_length,
    )
