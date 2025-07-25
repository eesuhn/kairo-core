import uvicorn
import socket

from contextlib import closing
from typing import Optional


def find_free_port(start_port: int = 8000, max_port: int = 8010) -> Optional[int]:
    for port in range(start_port, max_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return None


def main() -> None:
    port = find_free_port()

    if port is None:
        print("No free ports available in range 8000-8010")
        return

    print(f"Starting API server at http://127.0.0.1:{port}")

    try:
        uvicorn.run(
            "api.__main__:app",
            host="127.0.0.1",
            port=port,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user.")


if __name__ == "__main__":
    main()
