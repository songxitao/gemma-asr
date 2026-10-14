"""Test basic environment, dependencies and project package importability."""

import torch
import sentencepiece  # noqa: F401
import pyarrow as pa
from pypinyin import Style, pinyin


def test_package_imports():
    """Verify core packages can be imported without error."""
    import gemma_asr
    import gemma_asr.features
    import gemma_asr.model
    import gemma_asr.scoring
    import gemma_asr.text
    import gemma_asr.units

    assert gemma_asr is not None


def test_torch_and_cuda():
    """Verify PyTorch version and CUDA accessibility."""
    assert torch.__version__.startswith("2.5")
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    x = torch.ones(2, 2, device=device)
    assert (x + x).sum().item() == 8.0


def test_pyarrow_basic():
    """Verify pyarrow table and schema creation with numpy 1.x compatibility."""
    schema = pa.schema([("text", pa.string()), ("duration", pa.float32())])
    table = pa.Table.from_arrays([pa.array(["测试"]), pa.array([1.5], type=pa.float32())], schema=schema)
    assert table.num_rows == 1
    assert table.column_names == ["text", "duration"]


def test_pypinyin_basic():
    """Verify pypinyin functionality for P-CER calculation."""
    res = pinyin("人工智能语音", style=Style.NORMAL)
    flat = [item[0] for item in res]
    assert len(flat) == 6
    assert flat == ["ren", "gong", "zhi", "neng", "yu", "yin"]
