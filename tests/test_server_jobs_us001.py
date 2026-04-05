"""Tests for server/jobs.py — US-001: Pydantic job types."""

import pytest
from pydantic import BaseModel

from server.jobs import JobData, JobResult, PythonProgress


# AC01 — all three classes are exported as Pydantic BaseModel subclasses
class TestAC01Exports:
    def test_job_data_is_base_model(self):
        assert issubclass(JobData, BaseModel)

    def test_job_result_is_base_model(self):
        assert issubclass(JobResult, BaseModel)

    def test_python_progress_is_base_model(self):
        assert issubclass(PythonProgress, BaseModel)


# AC02 — JobData fields
class TestAC02JobData:
    def _valid(self):
        return dict(
            action="run",
            media="image",
            model="sdxl",
            script="pipelines/image/sdxl/t2i.py",
            args={"prompt": "cat"},
            script_base="/opt/scripts",
            uv_path="/usr/bin/uv",
        )

    def test_job_data_instantiates(self):
        obj = JobData(**self._valid())
        assert obj.action == "run"
        assert obj.media == "image"
        assert obj.model == "sdxl"
        assert obj.script == "pipelines/image/sdxl/t2i.py"
        assert obj.args == {"prompt": "cat"}
        assert obj.script_base == "/opt/scripts"
        assert obj.uv_path == "/usr/bin/uv"

    def test_action_is_str(self):
        fields = JobData.model_fields
        assert fields["action"].annotation is str

    def test_media_is_str(self):
        assert JobData.model_fields["media"].annotation is str

    def test_model_is_str(self):
        assert JobData.model_fields["model"].annotation is str

    def test_script_is_str(self):
        assert JobData.model_fields["script"].annotation is str

    def test_args_is_dict(self):
        assert JobData.model_fields["args"].annotation is dict

    def test_script_base_is_str(self):
        assert JobData.model_fields["script_base"].annotation is str

    def test_uv_path_is_str(self):
        assert JobData.model_fields["uv_path"].annotation is str

    def test_missing_required_field_raises(self):
        data = self._valid()
        del data["action"]
        with pytest.raises(Exception):
            JobData(**data)


# AC03 — JobResult fields
class TestAC03JobResult:
    def test_job_result_instantiates(self):
        obj = JobResult(output_path="/tmp/out.png")
        assert obj.output_path == "/tmp/out.png"

    def test_output_path_is_str(self):
        assert JobResult.model_fields["output_path"].annotation is str

    def test_missing_output_path_raises(self):
        with pytest.raises(Exception):
            JobResult()


# AC04 — PythonProgress fields
class TestAC04PythonProgress:
    def test_required_fields_only(self):
        obj = PythonProgress(step="loading", pct=0.5)
        assert obj.step == "loading"
        assert obj.pct == 0.5
        assert obj.frame is None
        assert obj.total is None
        assert obj.output is None
        assert obj.error is None

    def test_all_fields(self):
        obj = PythonProgress(
            step="sampling",
            pct=0.75,
            frame=10,
            total=50,
            output="frame_010.png",
            error=None,
        )
        assert obj.frame == 10
        assert obj.total == 50
        assert obj.output == "frame_010.png"

    def test_step_is_str(self):
        assert PythonProgress.model_fields["step"].annotation is str

    def test_pct_is_float(self):
        assert PythonProgress.model_fields["pct"].annotation is float

    def test_frame_is_optional_int(self):
        import typing
        hint = PythonProgress.model_fields["frame"].annotation
        # Should accept None (optional field)
        obj = PythonProgress(step="s", pct=0.0, frame=None)
        assert obj.frame is None
        obj2 = PythonProgress(step="s", pct=0.0, frame=5)
        assert obj2.frame == 5

    def test_total_is_optional_int(self):
        obj = PythonProgress(step="s", pct=0.0, total=None)
        assert obj.total is None
        obj2 = PythonProgress(step="s", pct=0.0, total=100)
        assert obj2.total == 100

    def test_output_is_optional_str(self):
        obj = PythonProgress(step="s", pct=0.0, output=None)
        assert obj.output is None
        obj2 = PythonProgress(step="s", pct=0.0, output="done.png")
        assert obj2.output == "done.png"

    def test_error_is_optional_str(self):
        obj = PythonProgress(step="s", pct=0.0, error=None)
        assert obj.error is None
        obj2 = PythonProgress(step="s", pct=0.0, error="OOM")
        assert obj2.error == "OOM"

    def test_missing_required_step_raises(self):
        with pytest.raises(Exception):
            PythonProgress(pct=0.5)

    def test_missing_required_pct_raises(self):
        with pytest.raises(Exception):
            PythonProgress(step="s")
