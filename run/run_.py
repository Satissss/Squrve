from core.base import Router
from core.engine import Engine

if __name__ == "__main__":
    router = Router(config_path="...")

    engine = Engine(router)

    engine.execute()
