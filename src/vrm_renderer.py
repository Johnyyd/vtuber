"""
VRM Renderer using pygltflib (no pyvrm dependency).
VRM is a glTF extension; pygltflib loads the binary .vrm file and exposes the VRM extension.
"""

from pygltflib import GLTF2
from typing import Dict, List, Optional


class VrModel:
    def __init__(self, vrm_path: str):
        self.vrm_path = vrm_path
        self.gltf = GLTF2().load_binary(vrm_path)
        self._blendshape_names: List[str] = []
        self._extract_blendshape_names()

    def _extract_blendshape_names(self):
        """Extract blendshape group names from VRM extension."""
        if (
            hasattr(self.gltf, "extensions")
            and "VRM" in self.gltf.extensions
            and "blendShapeMaster" in self.gltf.extensions["VRM"]
        ):
            bsm = self.gltf.extensions["VRM"]["blendShapeMaster"]
            if "blendShapeGroups" in bsm:
                self._blendshape_names = [
                    group.get("presetName") or group.get("name")
                    for group in bsm["blendShapeGroups"]
                ]

    @property
    def blendshape_names(self) -> List[str]:
        return self._blendshape_names

    def set_blendshapes(self, blendshapes: Dict[str, float]):
        """Apply blendshape values (for debugging / printing)."""
        for name, value in blendshapes.items():
            print(f"  [{name}] -> {value:.3f}")

    def render(self):
        """
        Placeholder for actual rendering.
        Replace with your renderer (Unity, Three.js, PyOpenGL, Godot, etc.)
        """
        pass


def render_vrm(blendshapes: Dict[str, float]):
    vrm_path = "resource/character.vrm"
    vr = VrModel(vrm_path)
    vr.set_blendshapes(blendshapes)
    vr.render()