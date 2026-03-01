"""Tests for pbzarr._backends — Backend enum."""

import pytest

from pbzarr import Backend


class TestBackendEnum:
    def test_numpy_value(self):
        assert Backend.NUMPY.value == "numpy"

    def test_dask_value(self):
        assert Backend.DASK.value == "dask"

    def test_from_value_string(self):
        assert Backend.from_value("numpy") is Backend.NUMPY
        assert Backend.from_value("dask") is Backend.DASK

    def test_from_value_enum(self):
        assert Backend.from_value(Backend.NUMPY) is Backend.NUMPY

    def test_from_value_invalid(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Backend.from_value("cupy")

    def test_from_value_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Backend.from_value(42)  # type: ignore[arg-type]
