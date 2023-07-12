from __future__ import annotations
import sys
from anki import _rsbridge, backend_pb2                          # noqa: E402
from anki._backend_generated import RustBackendGenerated         # noqa: E402


class RustBackend(RustBackendGenerated):
    """
    Python bindings for Anki's Rust libraries.

    Please do not access methods on the backend directly - they may be changed
    or removed at any time. Instead, please use the methods on the collection
    instead. Eg, don't use col._backend.all_deck_config(), instead use
    col.decks.all_config()

    If you need to access a backend method that is not currently accessible
    via the collection, please send through a pull request that adds a
    public method.
    """

    @staticmethod
    def initialize_logging(path: str | None = None) -> None:
        _rsbridge.initialize_logging(path)

    def __init__(
        self,
        langs: list[str] | None = None,
        server: bool = False,
    ) -> None:
        # pick up global defaults if not provided
        # import anki.lang

        # if langs is None:
        #     langs = [anki.lang.current_lang]

        init_msg = backend_pb2.BackendInit(
            preferred_langs=langs,
            server=server,
        )
        self._backend = _rsbridge.open_backend(init_msg.SerializeToString())

    def _run_command(self, service: int, method: int, input: bytes) -> bytes:
        try:
            return self._backend.command(service, method, input)
        except Exception as error:
            error_bytes = bytes(error.args[0])

        err = backend_pb2.BackendError()
        err.ParseFromString(error_bytes)
        raise backend_exception_to_pylib(err)


def backend_exception_to_pylib(err: backend_pb2.BackendError) -> Exception:
    class AnkiException(Exception):
        pass

    class BackendError(AnkiException):
        def __init__(self, message: str) -> None:
            super().__init__()
            self._message = message

        def __str__(self) -> str:
            return self._message

    return BackendError(err.message)
