try:
    import pyvrm
except Exception:
    pyvrm = None


class VrModel:
    def __init__(self, vrm_path: str):
        self.vrm_path = vrm_path
        self.model = pyvrm.load_vrm(vrm_path) if pyvrm else None

    def set_blendshapes(self, blendshapes: dict):
        if self.model is None:
            print("[VRM] Blendshapes:", blendshapes)
            return
        for name, value in blendshapes.items():
            print(f"  [{name}] -> {value:.3f}")

    def render(self):
        pass


def render_vrm(blendshapes: dict):
    vrm_path = "/resource/character.vrm"
    vr = VrModel(vrm_path)
    vr.set_blendshapes(blendshapes)
    vr.render()