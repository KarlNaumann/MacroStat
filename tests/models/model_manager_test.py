"""
pytest code for the Macrostat Model Manager module
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import os
from unittest.mock import MagicMock, patch

import pytest

from macrostat.models.model_manager import (
    ModelClasses,
    get_available_models,
    get_model,
    get_model_classes,
)


class TestModelManager:
    """Tests for the Model Manager module."""

    def setup_test_models(self, tmp_path, monkeypatch):
        """Set up test model directory structure."""

        # Create the __init__.py in the test directory itself
        init_file = os.path.join(str(tmp_path), "__init__.py")
        with open(init_file, "w") as f:
            f.write("")  # Create empty file

        # Ensure the directory exists and is accessible
        assert os.path.exists(
            init_file
        ), f"Test directory not properly set up at {init_file}"

        # Create model directories and files
        for model_name in ["ModelA", "ModelB"]:
            model_dir = os.path.join(str(tmp_path), model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Create required files
            files = [
                "__init__.py",
                "parameters.py",
                "variables.py",
                "scenarios.py",
                "behavior.py",
                f"{model_name.lower()}.py",
            ]

            for file in files:
                file_path = os.path.join(model_dir, file)
                with open(file_path, "w") as f:
                    f.write("")  # Create empty file'

        monkeypatch.setattr(
            "macrostat.models.model_manager.__file__",
            str(tmp_path / "model_manager.py"),
        )
        monkeypatch.chdir(tmp_path)

        return tmp_path

    def test_get_available_models(self, tmp_path, monkeypatch):
        """Test getting available models."""
        # Set up test directory structure
        self.setup_test_models(tmp_path, monkeypatch)

        models = get_available_models(model_directory=os.getcwd())
        assert set(models) == {"ModelA", "ModelB"}

    def test_get_model_invalid_model(self, tmp_path, monkeypatch):
        """Test getting an invalid model."""
        self.setup_test_models(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as exc_info:
            get_model("InvalidModel", model_directory=os.getcwd())
        assert "Invalid or unavailable model" in str(exc_info.value)

    def test_get_model_success(self, tmp_path, monkeypatch):
        """Test successful retrieval of a model."""
        # Set up test directory structure
        self.setup_test_models(tmp_path, monkeypatch)

        # Create mock module content
        mock_class = MagicMock()

        def mock_import(modulename, fromlist, *args, **kwargs):
            """Mock import function."""
            mock_module = MagicMock()
            setattr(mock_module, "ModelA", mock_class)
            return mock_module

        with patch("builtins.__import__", side_effect=mock_import):
            result = get_model("ModelA", model_directory=os.getcwd())
            assert result == mock_class

    def test_get_model_import_error(self, tmp_path, monkeypatch):
        """Test import error retrieval of a model."""
        # Set up test directory structure
        self.setup_test_models(tmp_path, monkeypatch)

        with pytest.raises(ImportError) as exc_info:
            get_model("ModelA", model_directory=os.getcwd())
        assert "Could not import model ModelA" in str(exc_info.value)

    def test_get_model_classes_invalid_model(self, tmp_path, monkeypatch):
        """Test getting classes for an invalid model."""
        self.setup_test_models(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as exc_info:
            get_model_classes("InvalidModel", model_directory=os.getcwd())
            assert "Invalid or unavailable model" in str(exc_info.value)

    def test_get_model_classes_success(self, tmp_path, monkeypatch):
        """Test successful retrieval of model classes."""
        # Set up test directory structure
        self.setup_test_models(tmp_path, monkeypatch)

        # Create mock module content
        mock_classes = {
            "Model": MagicMock(),
            "Modela": MagicMock(),
            "Modelb": MagicMock(),
            "Behavior": MagicMock(),
            "Parameters": MagicMock(),
            "Variables": MagicMock(),
            "Scenarios": MagicMock(),
        }

        def mock_import(modulename, fromlist, *args, **kwargs):
            """Mock import function."""
            name = fromlist[0]
            clsname = modulename.split(".")[-1].capitalize()
            cls = mock_classes[clsname]
            mock_module = MagicMock()
            setattr(mock_module, name, cls)
            return mock_module

        with patch("builtins.__import__", side_effect=mock_import):
            result = get_model_classes("ModelA", model_directory=os.getcwd())

            assert isinstance(result, ModelClasses)
            assert result.Behavior == mock_classes["Behavior"]
            assert result.Parameters == mock_classes["Parameters"]
            assert result.Variables == mock_classes["Variables"]
            assert result.Scenarios == mock_classes["Scenarios"]

    def test_get_model_classes_import_error(self, tmp_path, monkeypatch):
        """Test import error retrieval of model classes."""
        # Set up test directory structure
        self.setup_test_models(tmp_path, monkeypatch)
        with pytest.raises(ImportError) as exc_info:
            get_model_classes("ModelA", model_directory=os.getcwd())
        assert "Could not import model ModelA" in str(exc_info.value)
