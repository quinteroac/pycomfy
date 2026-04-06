"""Tests for server REST endpoints — US-001 it_000044: Submit inference jobs via REST.

Covers:
  AC01 — POST /jobs/create/image accepts model/prompt/optional fields, returns {job_id, status: queued}
  AC02 — POST /jobs/create/video accepts model/prompt/input
  AC03 — POST /jobs/create/audio accepts model/prompt/optional audio fields
  AC04 — POST /jobs/edit/image accepts model/prompt/input (required)
  AC05 — POST /jobs/upscale/image accepts model/prompt/input (required)
  AC06 — All endpoints call submit_job() and return job ID in <200ms
  AC07 — All request bodies validated by Pydantic models in server/schemas.py
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError
from starlette.testclient import TestClient

from server.app import app
from server.schemas import (
    CreateAudioRequest,
    CreateImageRequest,
    CreateVideoRequest,
    EditImageRequest,
    JobResponse,
    UpscaleImageRequest,
)

_FAKE_JOB_ID = "test-job-id-1234"
_SUBMIT_PATCH = "server.gateway.submit_job"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_submit():
    with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID) as mock:
        yield mock


# ---------------------------------------------------------------------------
# AC07 — Pydantic schemas exported from server/schemas.py
# ---------------------------------------------------------------------------

class TestAC07Schemas:
    def test_all_schema_classes_are_base_models(self):
        for cls in (
            CreateImageRequest,
            CreateVideoRequest,
            CreateAudioRequest,
            EditImageRequest,
            UpscaleImageRequest,
            JobResponse,
        ):
            assert issubclass(cls, BaseModel), f"{cls.__name__} must be a Pydantic BaseModel"

    def test_create_image_requires_model_and_prompt(self):
        obj = CreateImageRequest(model="sdxl", prompt="a cat")
        assert obj.model == "sdxl"
        assert obj.prompt == "a cat"

    def test_create_image_optional_fields_default_none(self):
        obj = CreateImageRequest(model="m", prompt="p")
        assert obj.width is None
        assert obj.height is None
        assert obj.steps is None
        assert obj.cfg is None
        assert obj.seed is None
        assert obj.negative_prompt is None
        assert obj.input is None

    def test_create_image_accepts_optional_fields(self):
        obj = CreateImageRequest(
            model="sdxl", prompt="p", width=512, height=512, steps=20,
            cfg=7.0, seed=42, negative_prompt="blur"
        )
        assert obj.width == 512
        assert obj.cfg == 7.0

    def test_create_image_missing_model_raises(self):
        with pytest.raises(ValidationError):
            CreateImageRequest(prompt="p")  # type: ignore[call-arg]

    def test_create_image_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            CreateImageRequest(model="sdxl")  # type: ignore[call-arg]

    def test_create_video_requires_model_and_prompt(self):
        obj = CreateVideoRequest(model="ltx2", prompt="a wave")
        assert obj.model == "ltx2"
        assert obj.input is None

    def test_create_video_accepts_input(self):
        obj = CreateVideoRequest(model="ltx2", prompt="p", input="/tmp/frame.png")
        assert obj.input == "/tmp/frame.png"

    def test_create_audio_requires_model_and_prompt(self):
        obj = CreateAudioRequest(model="ace_step", prompt="music")
        assert obj.model == "ace_step"
        assert obj.lyrics is None
        assert obj.bpm is None

    def test_create_audio_accepts_optional_fields(self):
        obj = CreateAudioRequest(model="m", prompt="p", lyrics="la la", bpm=120, length=30.0)
        assert obj.lyrics == "la la"
        assert obj.bpm == 120
        assert obj.length == 30.0

    def test_edit_image_requires_input(self):
        obj = EditImageRequest(model="qwen", prompt="fix", input="/img/in.png")
        assert obj.input == "/img/in.png"

    def test_edit_image_missing_input_raises(self):
        with pytest.raises(ValidationError):
            EditImageRequest(model="qwen", prompt="fix")  # type: ignore[call-arg]

    def test_upscale_image_requires_input(self):
        obj = UpscaleImageRequest(model="esrgan", prompt="upscale", input="/img/in.png")
        assert obj.input == "/img/in.png"

    def test_upscale_image_missing_input_raises(self):
        with pytest.raises(ValidationError):
            UpscaleImageRequest(model="esrgan", prompt="upscale")  # type: ignore[call-arg]

    def test_job_response_fields(self):
        obj = JobResponse(job_id="abc-123", status="queued")
        assert obj.job_id == "abc-123"
        assert obj.status == "queued"


# ---------------------------------------------------------------------------
# AC01 — POST /jobs/create/image
# ---------------------------------------------------------------------------

class TestAC01CreateImage:
    def test_returns_job_id_and_queued_status(self, client, mock_submit):
        resp = client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "a cat"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == _FAKE_JOB_ID
        assert body["status"] == "queued"

    def test_accepts_optional_generation_fields(self, client, mock_submit):
        resp = client.post("/jobs/create/image", json={
            "model": "sdxl", "prompt": "cat", "width": 1024, "height": 1024,
            "steps": 30, "cfg": 7.5, "seed": 1, "negative_prompt": "blur"
        })
        assert resp.status_code == 200

    def test_missing_model_returns_422(self, client):
        resp = client.post("/jobs/create/image", json={"prompt": "cat"})
        assert resp.status_code == 422

    def test_missing_prompt_returns_422(self, client):
        resp = client.post("/jobs/create/image", json={"model": "sdxl"})
        assert resp.status_code == 422

    def test_responds_within_200ms(self, client, mock_submit):
        start = time.perf_counter()
        client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "test"})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Took {elapsed_ms:.1f}ms (limit: 200ms)"


# ---------------------------------------------------------------------------
# AC02 — POST /jobs/create/video
# ---------------------------------------------------------------------------

class TestAC02CreateVideo:
    def test_returns_job_id_and_queued_status(self, client, mock_submit):
        resp = client.post("/jobs/create/video", json={"model": "ltx2", "prompt": "wave"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == _FAKE_JOB_ID
        assert body["status"] == "queued"

    def test_accepts_optional_input_for_i2v(self, client, mock_submit):
        resp = client.post("/jobs/create/video", json={
            "model": "ltx2", "prompt": "wave", "input": "/tmp/frame.png"
        })
        assert resp.status_code == 200

    def test_missing_model_returns_422(self, client):
        resp = client.post("/jobs/create/video", json={"prompt": "wave"})
        assert resp.status_code == 422

    def test_responds_within_200ms(self, client, mock_submit):
        start = time.perf_counter()
        client.post("/jobs/create/video", json={"model": "ltx2", "prompt": "wave"})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Took {elapsed_ms:.1f}ms (limit: 200ms)"


# ---------------------------------------------------------------------------
# AC03 — POST /jobs/create/audio
# ---------------------------------------------------------------------------

class TestAC03CreateAudio:
    def test_returns_job_id_and_queued_status(self, client, mock_submit):
        resp = client.post("/jobs/create/audio", json={"model": "ace_step", "prompt": "jazz"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == _FAKE_JOB_ID
        assert body["status"] == "queued"

    def test_accepts_optional_audio_fields(self, client, mock_submit):
        resp = client.post("/jobs/create/audio", json={
            "model": "ace_step", "prompt": "jazz", "lyrics": "la la", "bpm": 120,
            "length": 30.0, "steps": 60
        })
        assert resp.status_code == 200

    def test_missing_model_returns_422(self, client):
        resp = client.post("/jobs/create/audio", json={"prompt": "jazz"})
        assert resp.status_code == 422

    def test_responds_within_200ms(self, client, mock_submit):
        start = time.perf_counter()
        client.post("/jobs/create/audio", json={"model": "ace_step", "prompt": "jazz"})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Took {elapsed_ms:.1f}ms (limit: 200ms)"


# ---------------------------------------------------------------------------
# AC04 — POST /jobs/edit/image
# ---------------------------------------------------------------------------

class TestAC04EditImage:
    def test_returns_job_id_and_queued_status(self, client, mock_submit):
        resp = client.post("/jobs/edit/image", json={
            "model": "qwen", "prompt": "fix colors", "input": "/tmp/in.png"
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == _FAKE_JOB_ID
        assert body["status"] == "queued"

    def test_missing_input_returns_422(self, client):
        resp = client.post("/jobs/edit/image", json={"model": "qwen", "prompt": "fix"})
        assert resp.status_code == 422

    def test_missing_model_returns_422(self, client):
        resp = client.post("/jobs/edit/image", json={"prompt": "fix", "input": "/in.png"})
        assert resp.status_code == 422

    def test_responds_within_200ms(self, client, mock_submit):
        start = time.perf_counter()
        client.post("/jobs/edit/image", json={
            "model": "qwen", "prompt": "fix", "input": "/tmp/in.png"
        })
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Took {elapsed_ms:.1f}ms (limit: 200ms)"


# ---------------------------------------------------------------------------
# AC05 — POST /jobs/upscale/image
# ---------------------------------------------------------------------------

class TestAC05UpscaleImage:
    def test_returns_job_id_and_queued_status(self, client, mock_submit):
        resp = client.post("/jobs/upscale/image", json={
            "model": "esrgan", "prompt": "upscale 4x", "input": "/tmp/in.png"
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == _FAKE_JOB_ID
        assert body["status"] == "queued"

    def test_missing_input_returns_422(self, client):
        resp = client.post("/jobs/upscale/image", json={"model": "esrgan", "prompt": "upscale"})
        assert resp.status_code == 422

    def test_responds_within_200ms(self, client, mock_submit):
        start = time.perf_counter()
        client.post("/jobs/upscale/image", json={
            "model": "esrgan", "prompt": "4x", "input": "/tmp/in.png"
        })
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Took {elapsed_ms:.1f}ms (limit: 200ms)"


# ---------------------------------------------------------------------------
# AC06 — All endpoints call submit_job() from server/submit.py
# ---------------------------------------------------------------------------

class TestAC06SubmitJobCalled:
    def test_create_image_calls_submit_job(self, client, mock_submit):
        client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "cat"})
        mock_submit.assert_called_once()

    def test_create_video_calls_submit_job(self, client, mock_submit):
        client.post("/jobs/create/video", json={"model": "ltx2", "prompt": "wave"})
        mock_submit.assert_called_once()

    def test_create_audio_calls_submit_job(self, client, mock_submit):
        client.post("/jobs/create/audio", json={"model": "ace_step", "prompt": "jazz"})
        mock_submit.assert_called_once()

    def test_edit_image_calls_submit_job(self, client, mock_submit):
        client.post("/jobs/edit/image", json={
            "model": "qwen", "prompt": "fix", "input": "/in.png"
        })
        mock_submit.assert_called_once()

    def test_upscale_image_calls_submit_job(self, client, mock_submit):
        client.post("/jobs/upscale/image", json={
            "model": "esrgan", "prompt": "4x", "input": "/in.png"
        })
        mock_submit.assert_called_once()

    def test_submit_job_receives_job_data_with_correct_model(self, client, mock_submit):
        from server.jobs import JobData

        client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "cat"})
        args, _ = mock_submit.call_args
        assert isinstance(args[0], JobData)
        assert args[0].model == "sdxl"

    def test_submit_job_media_matches_endpoint(self, client, mock_submit):
        client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "cat"})
        args, _ = mock_submit.call_args
        assert args[0].media == "image"

    def test_create_video_media_is_video(self, client, mock_submit):
        client.post("/jobs/create/video", json={"model": "ltx2", "prompt": "wave"})
        args, _ = mock_submit.call_args
        assert args[0].media == "video"

    def test_create_audio_media_is_audio(self, client, mock_submit):
        client.post("/jobs/create/audio", json={"model": "ace_step", "prompt": "jazz"})
        args, _ = mock_submit.call_args
        assert args[0].media == "audio"

    def test_create_image_action_is_create(self, client, mock_submit):
        client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "cat"})
        args, _ = mock_submit.call_args
        assert args[0].action == "create"

    def test_edit_image_action_is_edit(self, client, mock_submit):
        client.post("/jobs/edit/image", json={
            "model": "qwen", "prompt": "fix", "input": "/in.png"
        })
        args, _ = mock_submit.call_args
        assert args[0].action == "edit"

    def test_upscale_image_action_is_upscale(self, client, mock_submit):
        client.post("/jobs/upscale/image", json={
            "model": "esrgan", "prompt": "4x", "input": "/in.png"
        })
        args, _ = mock_submit.call_args
        assert args[0].action == "upscale"

    def test_returned_job_id_matches_submit_job_return(self, client, mock_submit):
        resp = client.post("/jobs/create/image", json={"model": "sdxl", "prompt": "cat"})
        assert resp.json()["job_id"] == _FAKE_JOB_ID
