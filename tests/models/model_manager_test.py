"""
pytest code for the Macrostat Model Manager module
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

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

    def setup_test_models(self, tmp_path):
        """Set up test model directory structure."""
        # Create model directories and files
        for model_name in ["ModelA", "ModelB"]:
            model_dir = tmp_path / model_name
            model_dir.mkdir()

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
                (model_dir / file).touch()

        return tmp_path

    def test_get_available_models(self, tmp_path):
        """Test getting available models."""
        # Set up test directory structure
        test_dir = self.setup_test_models(tmp_path)

        # Patch the model manager's directory to use our test directory
        with patch(
            "macrostat.models.model_manager.__file__", str(test_dir / "__init__.py")
        ):
            models = get_available_models()
            assert set(models) == {"ModelA", "ModelB"}

    def test_get_model_invalid_model(self):
        """Test getting an invalid model."""
        with pytest.raises(ValueError) as exc_info:
            get_model("InvalidModel")
        assert "Invalid or unavailable model" in str(exc_info.value)

    def test_get_model_success(self, tmp_path):
        """Test successful retrieval of a model."""
        # Set up test directory structure
        test_dir = self.setup_test_models(tmp_path)

        # Create mock module content
        mock_class = MagicMock()

        def mock_import(modulename, fromlist, *args, **kwargs):
            """Mock import function."""
            mock_module = MagicMock()
            setattr(mock_module, "ModelA", mock_class)
            return mock_module

        with patch(
            "macrostat.models.model_manager.__file__", str(test_dir / "__init__.py")
        ), patch("builtins.__import__", side_effect=mock_import):
            result = get_model("ModelA")
            assert result == mock_class

    def test_get_model_import_error(self, tmp_path):
        """Test import error retrieval of a model."""
        # Set up test directory structure
        test_dir = self.setup_test_models(tmp_path)

        with patch(
            "macrostat.models.model_manager.__file__", str(test_dir / "__init__.py")
        ):
            with pytest.raises(ImportError) as exc_info:
                get_model("ModelA")
            assert "Could not import model ModelA" in str(exc_info.value)

    def test_get_model_classes_invalid_model(self):
        """Test getting classes for an invalid model."""
        with pytest.raises(ValueError) as exc_info:
            get_model_classes("InvalidModel")
        assert "Invalid or unavailable model" in str(exc_info.value)

    def test_get_model_classes_success(self, tmp_path):
        """Test successful retrieval of model classes."""
        # Set up test directory structure
        test_dir = self.setup_test_models(tmp_path)

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

        with patch(
            "macrostat.models.model_manager.__file__", str(test_dir / "__init__.py")
        ), patch("builtins.__import__", side_effect=mock_import):
            result = get_model_classes("ModelA")

            assert isinstance(result, ModelClasses)
            assert result.Behavior == mock_classes["Behavior"]
            assert result.Parameters == mock_classes["Parameters"]
            assert result.Variables == mock_classes["Variables"]
            assert result.Scenarios == mock_classes["Scenarios"]

    def test_get_model_classes_import_error(self, tmp_path):
        """Test import error retrieval of model classes."""
        # Set up test directory structure
        test_dir = self.setup_test_models(tmp_path)
        with patch(
            "macrostat.models.model_manager.__file__", str(test_dir / "__init__.py")
        ):
            with pytest.raises(ImportError) as exc_info:
                get_model_classes("ModelA")
            assert "Could not import model ModelA" in str(exc_info.value)
