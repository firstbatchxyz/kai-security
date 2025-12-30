from pathlib import Path

import pytest  # type: ignore[import-not-found]

import kai.processes.adapter_chooser as adapter_chooser
from kai.processes.adapter_chooser import AdapterChooserProcess
from kai.schemas import AdapterChooserInput, Framework, Language, MasterContext


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_adapter_chooser_detects_rust_and_cmake_without_llm(
    tmp_path: Path, monkeypatch
):
    # Make this test deterministic (no network / no LLM).
    async def _no_llm(*args, **kwargs):
        raise RuntimeError("LLM disabled for unit test")

    monkeypatch.setattr(adapter_chooser, "get_model_response", _no_llm)
    monkeypatch.setattr(
        adapter_chooser,
        "get_model_pricing",
        lambda *a, **k: {"prompt": 0.0, "completion": 0.0},
    )

    # Arrange: a tiny fake repo with Cargo + CMake signals
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "0.1.0"\n')
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "lib.rs").write_text("pub fn foo() -> u32 { 1 }\n")

    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
    (tmp_path / "main.cpp").write_text("int main(){return 0;}\n")

    mc = MasterContext(root_path=str(tmp_path), compile_success=True, frameworks=[])
    proc = AdapterChooserProcess(mc)

    # Act: run with a bogus model name; deterministic detection should still work
    out = await proc.execute(
        AdapterChooserInput(model_name="__no_model__", use_openai=False)
    )

    # Assert
    assert out.success is True
    assert out.choice is not None
    assert Language.RUST in out.choice.languages
    assert Language.CPP in out.choice.languages
    assert Framework.CARGO in out.choice.frameworks
    assert Framework.CMAKE in out.choice.frameworks
