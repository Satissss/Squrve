import warnings

from core.task.multi.MultiTask import MultiTask


class SequenceTask(MultiTask):
    """ Task For Text-to-SQL """

    NAME = "SequenceTask"

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def run(self):
        if not self.tasks:
            warnings.warn(f"The `tasks` list is empty. Run is stopped. ", category=UserWarning)
            return
        for task_ in self.tasks:
            try:
                if not task_.is_end:
                    task_.run()
            except Exception as e:
                print(f"运行 {task_.task_id} 时发生错误，错误信息为 {e}")
