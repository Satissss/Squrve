import warnings
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Optional
from core.data_manage import save_dataset
from datetime import datetime


class Logger:
    def __init__(
            self,
            save_path: Union[str, PathLike] = None
    ):
        self._info = {}
        self._error_dataset = []
        self._content = ""
        self.save_path = save_path
        self.data_logger = {}

    def __getitem__(self, item):
        return self._info[item]

    def __setitem__(self, key, value):
        self._info[key] = value

    def set_by_dict(self, **kwargs):
        self._info.update(kwargs)

    def add_error_data(self, row: Dict):
        self._error_dataset.append(row)

    @property
    def is_errors_empty(self):
        return len(self._error_dataset) == 0

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    def info(self, string: str):
        self._content += f"{self._get_timestamp()} [INFO]  {string}\n"

    def warn(self, string: str):
        self._content += f"{self._get_timestamp()} [WARN]  {string}\n"

    def error(self, string: str):
        self._content += f"{self._get_timestamp()} [ERROR] {string}\n"

    def __str__(self):
        def _format_dict(d: dict, prefix: str = ""):
            lines = []
            for key, val in d.items():
                if not isinstance(val, (str, list, dict)) or not val:
                    continue
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(val, dict):
                    lines.append(f"-- {full_key}:\tvalue: {{")
                    lines.extend(_format_dict(val, prefix=f"\t{full_key}"))
                    lines.append(f"-- {full_key}:\tvalue: }}")
                else:
                    # todo 是否有必要保存所有中间结果
                    lines.append(f"-- {full_key}:\tvalue: {val}")
            return lines

        info_lines = ["[Key Item Value]"]
        info_lines.extend(_format_dict(self._info))
        info_lines.append("\n-------------------------------------------------------------------\n")
        info_lines.append("[Runtime Information]")
        info_lines.append(self._content.rstrip())
        return "\n".join(info_lines)

    def save(self, save_path: Union[str, PathLike] = None):
        if save_path is None:
            save_path = self.save_path

        if save_path is None:
            warnings.warn(f"The save_path is not available.", category=UserWarning)
            return

        if not self.is_errors_empty:
            error_save_path = Path(save_path).parent / "error_dataset.json"
            save_dataset(self._error_dataset, new_data_source=error_save_path)

        save_dataset(self.__str__(), new_data_source=save_path)

    def generate_data_logger(self, index: int | str):
        if index in self.data_logger:
            return self.data_logger[index]


        if self.save_path is None:
            warnings.warn("Task logger save_path is unavailable, cannot create data logger.",
                          category=UserWarning)
            save_path = None
        else:
            save_path = Path(self.save_path).parent / "data_log" / f"{index}.txt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

        data_logger = DataLogger(index, save_path)
        self.data_logger[index] = data_logger

        return data_logger

    def save_all_data_logs(self):
        for dl in self.data_logger.values():
            if dl is not None:
                dl.save()


class DataLogger:
    def __init__(
            self,
            index: Union[int, str],
            save_path: Union[str, PathLike, None] = None
    ):
        self._index = index
        self._save_path: Optional[Path] = Path(save_path) if save_path else None
        self._content: str = ""

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    def _append(self, string: str):
        print(string)
        self._content += string + "\n"

    def info(self, string: str):
        self._append(f"{self._get_timestamp()} [INFO]  {string}")

    def warn(self, string: str):
        self._append(f"{self._get_timestamp()} [WARN]  {string}")

    def error(self, string: str):
        self._append(f"{self._get_timestamp()} [ERROR] {string}")

    @property
    def index(self):
        return self._index

    @property
    def content(self) -> str:
        return self._content

    def clear(self):
        self._content = ""

    def save(self, save_path: Union[str, PathLike, None] = None):
        target_path = Path(save_path) if save_path else self._save_path
        if target_path is None:
            warnings.warn(f"The save_path is not available for DataLogger {self._index}.",
                          category=UserWarning)
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(self.content, new_data_source=target_path)
        self._save_path = target_path
