"""Import checks for the relocated scale-out drivers.

These modules pull the Morphling runtime through ``morphling.api``, so they
import only inside the Wasp Docker image (which is ``FROM`` the Morphling image
that carries the compiled runtime). Run under that image.
"""

import importlib

import pytest

MODULES = [
    "wasp.coordinator.run_multi_coordinator",
    "wasp.coordinator._multi_coordinator_training",
    "wasp.scaling.multi_coordinator_scaling_config",
    "wasp.scaling.export_selected_layouts",
    "wasp.traces.generate_trace",
]


@pytest.mark.parametrize("name", MODULES)
def test_driver_module_imports(name):
    importlib.import_module(name)
