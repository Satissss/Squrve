import hashlib
import re
from os import PathLike
from pathlib import Path
from typing import Optional, Union

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset
from core.utils import load_dataset, save_dataset
from squrve_bmsql.bmsql_backend import (
    BMSQLBackend,
    MockBMSQLBackend,
)
from squrve_bmsql.models import BMSQLGeneration, BMSQLRequest


_SAFE_FILENAME_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _safe_filename_component(value) -> str:
    text = str(value)
    if (
            text not in {".", ".."}
            and _SAFE_FILENAME_COMPONENT.fullmatch(text) is not None
    ):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"encoded-{digest}"


@BaseGenerator.register_actor
class BMSQLGenerator(BaseGenerator):
    NAME = "BMSQLGenerator"
    SKILL = "Run the original BiomedSQL BMSQL workflow as one Generator."

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            backend: Optional[BMSQLBackend] = None,
            backend_mode: Optional[str] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            **kwargs
    ):
        if backend is None:
            if backend_mode == "mock":
                backend = MockBMSQLBackend()
            else:
                raise ValueError(
                    "backend must be injected unless backend_mode is 'mock'"
                )
        self.dataset = dataset
        self.backend = backend
        self.is_save = is_save
        self.save_dir = save_dir

    def _resolve_schema(self, item, schema):
        if schema is None:
            schema = self.dataset.get_db_schema(item)
        if isinstance(schema, (str, PathLike)):
            schema = load_dataset(schema)
        if schema is None:
            raise ValueError("Failed to load a valid database schema for the sample")
        return schema

    def _store_generation(self, item, generation: BMSQLGeneration):
        for key, value in generation.to_dict().items():
            self.dataset.setitem(item, key, value)

        if not self.is_save or generation.pred_sql is None:
            return

        row = self.dataset[item]
        save_root = Path(self.save_dir)
        save_path = save_root
        if self.dataset.dataset_index is not None:
            save_path = save_path / _safe_filename_component(
                self.dataset.dataset_index
            )
        instance_component = _safe_filename_component(row["instance_id"])
        save_path = save_path / f"{self.NAME}_{instance_component}.sql"
        if not save_path.resolve().is_relative_to(save_root.resolve()):
            raise ValueError("BMSQL save destination must remain inside save_dir")
        save_dataset(generation.pred_sql, new_data_source=save_path)
        self.dataset.setitem(item, "pred_sql_path", str(save_path))

    def act(
            self,
            item,
            schema=None,
            schema_links=None,
            data_logger=None,
            **kwargs
    ):
        row = self.dataset[item]
        error_stage = "schema"
        try:
            resolved_schema = self._resolve_schema(item, schema)
            error_stage = "request"
            request = BMSQLRequest(
                instance_id=str(row["instance_id"]),
                question=str(row["question"]),
                schema=resolved_schema,
                domain_context=row.get("external") or row.get("domain_context"),
                metadata=dict(row.get("metadata") or {}),
            )
            error_stage = "backend"
            generation = self.backend.generate(request)
        except Exception as exc:
            generation = BMSQLGeneration.failure(
                str(exc),
                error_stage=error_stage,
            )
        self._store_generation(item, generation)
        return generation.pred_sql
