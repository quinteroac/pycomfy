"""Tests for US-009 — ltxv2_t2sv_example.py structure, pipeline coverage, and conventions.

Covers:
  AC01: file exists and follows the same structure / style as other examples
  AC02: all required pipeline steps are exercised (key function calls present)
  AC03: all required CLI args and env-var defaults are present
  AC04: script is written entirely in English (no Spanish text)
  AC05: file parses without syntax errors (typecheck / lint proxy)
"""

from __future__ import annotations

import ast
from pathlib import Path

EXAMPLE_PATH = (
    Path(__file__).parent.parent / "examples" / "ltxv2_t2sv_example.py"
)


def _source() -> str:
    return EXAMPLE_PATH.read_text(encoding="utf-8")


def _ast_tree() -> ast.Module:
    return ast.parse(_source(), filename=str(EXAMPLE_PATH))


# ---------------------------------------------------------------------------
# AC01 — file exists; follows same structural conventions as other examples
# ---------------------------------------------------------------------------


def test_example_file_exists() -> None:
    assert EXAMPLE_PATH.is_file(), f"Example file not found: {EXAMPLE_PATH}"


def test_example_has_shebang() -> None:
    first_line = _source().splitlines()[0]
    assert first_line.startswith("#!/usr/bin/env python"), (
        "Example must start with a shebang line"
    )


def test_example_has_module_docstring() -> None:
    tree = _ast_tree()
    docstring = ast.get_docstring(tree)
    assert docstring, "Example must have a module-level docstring"


def test_example_has_future_annotations_import() -> None:
    source = _source()
    assert "from __future__ import annotations" in source, (
        "Example must start with 'from __future__ import annotations'"
    )


def test_example_has_main_guard() -> None:
    source = _source()
    assert 'if __name__ == "__main__"' in source, (
        "Example must have a __main__ guard"
    )


def test_example_main_returns_int() -> None:
    """main() must exist and return int (verified by sys.exit(main()) pattern)."""
    source = _source()
    assert "sys.exit(main())" in source, (
        "Example must call sys.exit(main()) in the __main__ block"
    )


# ---------------------------------------------------------------------------
# AC02 — all required pipeline steps are exercised in order
# ---------------------------------------------------------------------------


def test_example_calls_check_runtime() -> None:
    assert "check_runtime" in _source()


def test_example_calls_load_unet() -> None:
    assert "load_unet(" in _source()


def test_example_calls_load_vae() -> None:
    assert "load_vae(" in _source()


def test_example_calls_load_ltxv_audio_vae() -> None:
    assert "load_ltxv_audio_vae(" in _source()


def test_example_calls_load_ltxav_text_encoder() -> None:
    assert "load_ltxav_text_encoder(" in _source()


def test_example_calls_generate_ltx2_prompt() -> None:
    assert "generate_ltx2_prompt(" in _source()


def test_example_calls_load_latent_upscale_model() -> None:
    assert "load_latent_upscale_model(" in _source()


def test_example_calls_encode_prompt() -> None:
    assert "encode_prompt(" in _source()


def test_example_calls_ltxv_conditioning() -> None:
    assert "ltxv_conditioning(" in _source()


def test_example_calls_ltxv_empty_latent_video() -> None:
    assert "ltxv_empty_latent_video(" in _source()


def test_example_calls_ltxv_crop_guides() -> None:
    assert "ltxv_crop_guides(" in _source()


def test_example_calls_ltxv_empty_latent_audio() -> None:
    assert "ltxv_empty_latent_audio(" in _source()


def test_example_calls_ltxv_concat_av_latent() -> None:
    assert "ltxv_concat_av_latent(" in _source()


def test_example_calls_sample() -> None:
    assert "sample(" in _source()


def test_example_calls_ltxv_separate_av_latent() -> None:
    assert "ltxv_separate_av_latent(" in _source()


def test_example_calls_ltxv_latent_upsample() -> None:
    assert "ltxv_latent_upsample(" in _source()


def test_example_calls_vae_decode_batch_tiled() -> None:
    assert "vae_decode_batch_tiled(" in _source()


def test_example_calls_ltxv_audio_vae_decode() -> None:
    assert "ltxv_audio_vae_decode(" in _source()


def test_example_saves_video_frames() -> None:
    source = _source()
    assert "_save_video_frames(" in source or "frame.save(" in source, (
        "Example must save video frames"
    )


def test_example_saves_audio() -> None:
    source = _source()
    assert "_save_audio(" in source or "torchaudio.save(" in source, (
        "Example must save audio output"
    )


# ---------------------------------------------------------------------------
# AC03 — all required CLI args and env-var defaults are present
# ---------------------------------------------------------------------------


def test_example_has_models_dir_arg() -> None:
    assert "--models-dir" in _source()


def test_example_has_pycomfy_models_dir_env() -> None:
    assert "PYCOMFY_MODELS_DIR" in _source()


def test_example_has_checkpoint_arg() -> None:
    assert "--checkpoint" in _source()


def test_example_has_pycomfy_checkpoint_env() -> None:
    assert "PYCOMFY_CHECKPOINT" in _source()


def test_example_has_unet_arg() -> None:
    assert '"--unet"' in _source() or "'--unet'" in _source()


def test_example_has_pycomfy_ltxv2_unet_env() -> None:
    assert "PYCOMFY_LTXV2_UNET" in _source()


def test_example_has_vae_arg() -> None:
    assert '"--vae"' in _source() or "'--vae'" in _source()


def test_example_has_pycomfy_ltxv2_vae_env() -> None:
    assert "PYCOMFY_LTXV2_VAE" in _source()


def test_example_has_audio_vae_arg() -> None:
    assert "--audio-vae" in _source()


def test_example_has_pycomfy_ltxv2_audio_vae_env() -> None:
    assert "PYCOMFY_LTXV2_AUDIO_VAE" in _source()


def test_example_has_text_encoder_arg() -> None:
    assert "--text-encoder" in _source()


def test_example_has_pycomfy_ltxv2_text_encoder_env() -> None:
    assert "PYCOMFY_LTXV2_TEXT_ENCODER" in _source()


def test_example_has_ltxav_checkpoint_arg() -> None:
    assert "--ltxav-checkpoint" in _source()


def test_example_has_pycomfy_ltxv2_ltxav_checkpoint_env() -> None:
    assert "PYCOMFY_LTXV2_LTXAV_CHECKPOINT" in _source()


def test_example_has_llm_arg() -> None:
    assert '"--llm"' in _source() or "'--llm'" in _source()


def test_example_has_pycomfy_llm_model_env() -> None:
    assert "PYCOMFY_LLM_MODEL" in _source()


def test_example_has_latent_upscale_model_arg() -> None:
    assert "--latent-upscale-model" in _source()


def test_example_has_pycomfy_ltxv2_latent_upscale_model_env() -> None:
    assert "PYCOMFY_LTXV2_LATENT_UPSCALE_MODEL" in _source()


def test_example_has_prompt_arg() -> None:
    assert "--prompt" in _source()


def test_example_has_negative_prompt_arg() -> None:
    assert "--negative-prompt" in _source()


def test_example_has_width_arg() -> None:
    assert "--width" in _source()


def test_example_has_height_arg() -> None:
    assert "--height" in _source()


def test_example_has_length_arg() -> None:
    assert "--length" in _source()


def test_example_has_frame_rate_arg() -> None:
    assert "--frame-rate" in _source()


def test_example_has_steps_arg() -> None:
    assert "--steps" in _source()


def test_example_has_cfg_arg() -> None:
    assert "--cfg" in _source()


def test_example_has_seed_arg() -> None:
    assert "--seed" in _source()


def test_example_has_sampler_arg() -> None:
    assert "--sampler" in _source()


def test_example_has_scheduler_arg() -> None:
    assert "--scheduler" in _source()


def test_example_has_output_dir_arg() -> None:
    assert "--output-dir" in _source()


# ---------------------------------------------------------------------------
# AC04 — script is written entirely in English
# ---------------------------------------------------------------------------

_SPANISH_WORDS = [
    "directorio",
    "modelo",
    "cargar",
    "guardar",
    "archivo",
    "ruta",
    "obligatorio",
    "pasos",
    "muestreo",
    "decodificar",
    "codificar",
    "vacío",
    "latente",
    "imagen",
    "video",  # "video" is English too, but "vídeo" is Spanish
]


def test_example_contains_no_spanish_words() -> None:
    source_lower = _source().lower()
    found = [word for word in _SPANISH_WORDS if word in source_lower]
    # "video" appears in both languages — allow it
    found = [w for w in found if w != "video"]
    assert not found, (
        f"Example must be written in English. Found Spanish-language words: {found}"
    )


# ---------------------------------------------------------------------------
# AC05 — file parses without syntax errors (typecheck / lint proxy)
# ---------------------------------------------------------------------------


def test_example_parses_without_syntax_errors() -> None:
    source = _source()
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))
    assert isinstance(tree, ast.Module)


def test_example_uses_lazy_imports() -> None:
    """comfy_diffusion imports must not appear at module top level — lazy pattern."""
    source = _source()
    lines = source.splitlines()
    # Find the first import of comfy_diffusion
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "comfy_diffusion" in stripped and stripped.startswith(
            ("import ", "from ")
        ):
            # It must be inside a function body (indented)
            assert line.startswith("    "), (
                f"comfy_diffusion import at line {i + 1} must be inside a function "
                "(lazy import pattern required)"
            )
            break
