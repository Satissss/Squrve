# Squrve

## Overview

**Squrve** is a lightweight and modular framework designed for rapid development and evaluation of `end-to-end` **Text-to-SQL** models. It integrates schema reduction, schema linking, and query generation into a flexible, configuration-driven pipeline. Users can quickly define and execute complex or concurrent tasks using JSON configuration files—no need to manually script task flows.

------

## 🧪 Sample Usage

```
from core.base import Router
from core.engine import Engine
import os

if __name__ == "__main__":
    router = Router(config_path="...")

    engine = Engine(router)

    engine.execute()
```

------

## 📌 Notes

- All modules can be instantiated and executed independently for debugging.
- JSON configs allow plug-and-play model integration and fast benchmarking.
- Parallel execution is supported via `ParallelTask`.