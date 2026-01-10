"""Tests for the colbert-export CLI.

Run with:
    pytest tests/test_cli.py -v

Or run specific tests:
    pytest tests/test_cli.py -v -k "test_help"
"""

import subprocess
import sys
from unittest.mock import patch

import pytest


class TestCLIHelp:
    """Test CLI help and basic invocation."""

    def test_export_help(self):
        """Test that colbert-export --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "colbert_export.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "colbert-export" in result.stdout
        assert "HuggingFace model name" in result.stdout
        assert "--quantize" in result.stdout
        assert "--force" in result.stdout

    def test_export_version(self):
        """Test that colbert-export --version works."""
        result = subprocess.run(
            [sys.executable, "-m", "colbert_export.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_quantize_help(self):
        """Test that colbert-quantize --help works."""
        from colbert_export.cli import quantize_main

        with patch("sys.argv", ["colbert-quantize", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                quantize_main()
            assert exc_info.value.code == 0


class TestExportFunction:
    """Test the export_model function."""

    def test_skip_existing_model(self, tmp_path):
        """Test that export skips when model.onnx already exists."""
        from colbert_export.export import export_model

        # Create a fake existing model
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("fake onnx model")

        # Should skip without calling pylate
        with patch("colbert_export.export.pylate_models") as mock_pylate:
            result = export_model(
                model_name="test/model",
                output_dir=model_dir,
                quantize=False,
                verbose=False,
            )

            # pylate should not be called since model exists
            mock_pylate.ColBERT.assert_not_called()
            assert result == model_dir

    def test_skip_existing_model_with_quantize(self, tmp_path):
        """Test that export quantizes when model exists but INT8 doesn't."""
        from colbert_export.export import export_model

        # Create a fake existing model without INT8
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("fake onnx model")

        # Should skip export but call quantize
        with patch("colbert_export.export.pylate_models") as mock_pylate:
            with patch("colbert_export.quantize.quantize_dynamic") as mock_quantize:
                result = export_model(
                    model_name="test/model",
                    output_dir=model_dir,
                    quantize=True,
                    verbose=False,
                )

                # pylate should not be called
                mock_pylate.ColBERT.assert_not_called()
                # quantize should be called
                mock_quantize.assert_called_once()
                assert result == model_dir

    def test_skip_existing_model_with_existing_int8(self, tmp_path):
        """Test that export skips both when model and INT8 exist."""
        from colbert_export.export import export_model

        # Create fake existing models
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("fake onnx model")
        (model_dir / "model_int8.onnx").write_text("fake int8 model")

        with patch("colbert_export.export.pylate_models") as mock_pylate:
            with patch("colbert_export.quantize.quantize_dynamic") as mock_quantize:
                result = export_model(
                    model_name="test/model",
                    output_dir=model_dir,
                    quantize=True,
                    verbose=False,
                )

                # Neither should be called
                mock_pylate.ColBERT.assert_not_called()
                mock_quantize.assert_not_called()
                assert result == model_dir

    def test_force_reexport(self, tmp_path):
        """Test that --force triggers re-export even when model exists."""
        from colbert_export.export import export_model

        # Create a fake existing model
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("fake onnx model")

        # With force=True, pylate should be called even though model exists
        # We just verify that it attempts to load the model (which will fail with mock)
        with patch("colbert_export.export.pylate_models.ColBERT") as mock_colbert:
            # Make it raise an error so we can verify it was called
            mock_colbert.side_effect = Exception("Mock: pylate was called")

            with pytest.raises(Exception, match="Mock: pylate was called"):
                export_model(
                    model_name="test/model",
                    output_dir=model_dir,
                    quantize=False,
                    verbose=False,
                    force=True,
                )

            # Verify pylate.ColBERT was called
            mock_colbert.assert_called_once()


class TestQuantizeFunction:
    """Test the quantize_model function."""

    def test_quantize_model(self, tmp_path):
        """Test quantize_model creates INT8 model."""
        from colbert_export.quantize import quantize_model

        # Create a fake model
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("fake onnx model")

        with patch("colbert_export.quantize.quantize_dynamic") as mock_quantize:
            result = quantize_model(model_dir, verbose=False)

            mock_quantize.assert_called_once()
            assert result == model_dir / "model_int8.onnx"

    def test_quantize_model_not_found(self, tmp_path):
        """Test quantize_model raises error when model.onnx not found."""
        from colbert_export.quantize import quantize_model

        model_dir = tmp_path / "nonexistent"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            quantize_model(model_dir, verbose=False)


class TestGetModelShortName:
    """Test the get_model_short_name helper."""

    def test_with_org(self):
        """Test extracting short name from org/model format."""
        from colbert_export.export import get_model_short_name

        assert get_model_short_name("lightonai/GTE-ModernColBERT-v1") == "GTE-ModernColBERT-v1"
        assert get_model_short_name("org/model-name") == "model-name"

    def test_without_org(self):
        """Test with just model name."""
        from colbert_export.export import get_model_short_name

        assert get_model_short_name("model-name") == "model-name"


class TestCLIIntegration:
    """Integration tests for CLI (these may be slow)."""

    @pytest.mark.slow
    def test_export_missing_model_error(self):
        """Test that exporting a non-existent model gives an error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "colbert_export.cli",
                "nonexistent/model-that-does-not-exist",
                "-o",
                "/tmp/test-nonexistent",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
