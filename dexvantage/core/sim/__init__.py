"""MuJoCo ground truth (spec §9.1): the project as a MuJoCo scene, and renders of it through each sensor's lens.

    scene.build_scene(project, variant=None, objects=True, lights=True) -> SceneBuild
    scene.save_scene(project, folder, variant=None) -> path of a self-contained MJCF (scene.xml + assets/)
    render.Renderer(project, variant=None, device='auto')   a context manager; .render(views), .info

See scene.py and render.py, docs/models/groundtruth.md. The study that compares these renders with the predictions
is core/studies/groundtruth.py.
"""
from .scene import SceneBuild, build_scene, save_scene  # noqa: F401
