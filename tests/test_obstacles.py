"""Project obstacles in the collision distances (core/robot/collision.py obstacle_distance, env_distance, obstacles),
their validation check, and the Robot page's status line."""
import numpy as np
import pytest

from conftest import pump


@pytest.fixture(scope='module')
def scripted(tmp_path_factory):
    from dexvantage.validation.checks_obstacles import obstacle_project
    return obstacle_project(str(tmp_path_factory.mktemp('obstacles')))


def _box_object(name, size, pos, quat=None):
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import TaskObject
    return TaskObject(name=name, shape='box', size=tuple(size),
                      pose=Pose(np.asarray(pos, float), np.asarray(quat if quat is not None else [0, 0, 0, 1.0])))


def test_obstacles_become_primitives(scripted):
    from dexvantage.core.robot import collision as C
    p, _uid, _ref, names = scripted
    obs, notes = C.obstacles(p)
    assert [o.name for o in obs] == names and notes == []
    by = {o.name: o for o in obs}
    assert by['box'].gtype == 'box' and by['box'].size == pytest.approx((0.10, 0.05, 0.07))     # half-edges
    assert by['ball'].gtype == 'sphere' and by['ball'].size == pytest.approx((0.06,))
    assert by['can'].gtype == 'cylinder' and by['can'].size == pytest.approx((0.04, 0.08))     # half-height
    assert by['crate'].gtype == 'mesh' and by['crate'].vertices.shape == (8, 3)
    t = by['table']
    assert t.owner == 'surface:table' and t.size == pytest.approx((0.5, 0.4, 0.02))
    assert t.position == pytest.approx([0.3, -0.1, 0.10])                                    # top at z = 0.12
    # MuJoCo's (w, x, y, z) order for a TaskObject pose given as (x, y, z, w)
    from dexvantage.core.frames import quat_to_mujoco
    o = next(x for x in p.workspace.objects if x.name == 'box')
    assert by['box'].quat_wxyz == pytest.approx(quat_to_mujoco(o.pose.quaternion))


def test_surface_without_thickness_uses_the_default_slab():
    from dexvantage.core.project import Project
    from dexvantage.core.robot import collision as C
    p = Project('t')
    p.workspace.surfaces = [{'uid': 's', 'name': 'Bench', 'z': 0.75, 'size': [1.0, 0.5]},
                            {'uid': 'bad', 'name': 'No size', 'z': 0.0, 'size': [0.0, 1.0]}]
    obs, _ = C.obstacles(p)
    assert len(obs) == 1
    assert obs[0].size[2] * 2 == pytest.approx(C.SURFACE_THICKNESS_M)
    assert obs[0].position[2] == pytest.approx(0.75 - C.SURFACE_THICKNESS_M / 2)


def test_flat_or_missing_meshes_are_left_out_with_a_note():
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import Project, TaskObject
    from dexvantage.core.robot import collision as C
    p = Project('t')
    p.meshes['plate'] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    p.workspace.objects = [TaskObject(name='Plate', shape='mesh', mesh_key='plate', pose=Pose()),
                           TaskObject(name='Ghost', shape='mesh', mesh_key='nothing', pose=Pose()),
                           TaskObject(name='Flat box', shape='box', size=(0.1, 0.0, 0.1), pose=Pose())]
    obs, notes = C.obstacles(p)
    assert obs == []
    assert len(notes) == 3 and 'Plate' in notes[0] and 'flat' in notes[0] and 'Flat box' in notes[2]


def test_env_distance_matches_closed_forms(scripted):
    from dexvantage.core.robot import collision as C
    from dexvantage.validation.checks_obstacles import C_OFF, R_ROBOT
    p, uid, ref, names = scripted
    B = p.robots[0].base.matrix()
    rng = np.random.default_rng(3)
    Q = rng.uniform(-0.35, 0.35, (60, 3))
    for q in Q:
        x = B[:3, :3] @ (C_OFF + q) + B[:3, 3]
        s = ref(x[None])[0] - R_ROBOT
        d, pair = C.env_distance(p, uid, q, distmax=1.0)
        assert d == pytest.approx(min(s.min(), 1.0), abs=1e-6)
        if np.sort(s)[1] - s.min() > 1e-3:
            assert pair[0] == 'sz' and pair[1] == names[int(np.argmin(s))]
            assert pair[2] == pytest.approx(d, abs=1e-12)


def test_obstacle_distance_flags_margin_and_caps_at_distmax(scripted):
    from dexvantage.core.robot import collision as C
    p, uid, ref, _ = scripted
    far = C.obstacle_distance(p, uid, [0.35, 0.35, 0.35], distmax=0.001)
    assert far.min_distance <= 0.001 and not far.in_collision
    near = C.obstacle_distance(p, uid, [0.0, 0.0, 0.0], margin=0.5, distmax=1.0)
    assert near.in_collision and near.pairs and all(dd < 0.5 for _a, _b, dd in near.pairs)
    assert [dd for _a, _b, dd in near.pairs] == sorted(dd for _a, _b, dd in near.pairs)


def test_robot_base_is_not_paired_with_the_table_it_stands_on():
    from dexvantage.core.robot import collision as C
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    for name in ('so100', 'ur5e'):
        p = sample_project(name)
        r = p.robots[0]
        d, pair = C.env_distance(p, r.uid)
        assert d > 0, (name, pair)
        base_links = {get_model(p, r.uid).link_names[0]}
        assert pair is None or pair[0] not in base_links


def test_an_object_on_the_wrist_is_a_collision():
    from dexvantage.core.robot import collision as C
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    w3 = m.link_pose(m.home, 'wrist_3_link')[:3, 3]
    p.workspace.objects.append(_box_object('Fixture', (0.05, 0.05, 0.05), w3))
    res = C.obstacle_distance(p, r.uid, m.home)
    assert res.in_collision and any(b == 'Fixture' for _a, b, _d in res.pairs)
    # move the arm away: the fixture is behind it, the distance is positive again
    q = m.home.copy()
    q[0] += 1.5
    d, pair = C.env_distance(p, r.uid, q)
    assert d > 0
    # a q of the wrong length is refused in plain words
    with pytest.raises(ValueError, match='moving joints'):
        C.env_distance(p, r.uid, [0.0, 1.0])


def test_obstacle_check_passes():
    from dexvantage.validation import suite as S
    S.load_checks()
    spec = next(s for s in S.CHECKS if s.id == 'kinematics.obstacle_distance')
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    try:
        c = S.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.error, c.exception, c.detail)
    assert c.detail['overlapping'] > 0 and c.detail['wrong_nearest_obstacle'] == 0


def test_robot_page_names_the_obstacle(app, settings):
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.mainwindow import MainWindow
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    w = MainWindow(p, soft_3d=True, settings=settings)
    w.resize(1400, 900)
    w.show()
    w.ctx.analysis.wait(5)
    pump(app, 0.05)
    try:
        w.open_page('robot')
        pump(app, 0.05)
        page = w.pages['robot']
        out = page.refresh_readouts()[r.uid]
        assert out['obstacles'][1] == 'clear' and 'Table (sample)' in out['obstacles'][2]
        assert page.head.status.text.text() == 'clear'
        assert 'or an obstacle' in page.head.explain.text()
        line = page._panels[r.uid].coll
        assert line.text.text().startswith('Collision · clear') and 'Obstacles:' in line.text.text()
        # an object at the wrist: the status line and the headline say so
        w3 = m.link_pose(m.home, 'wrist_3_link')[:3, 3]
        w.store.project.workspace.objects.append(_box_object('Fixture', (0.05, 0.05, 0.05), w3))
        out = page.refresh_readouts()[r.uid]
        assert out['collision'][0] == 'critical' and 'Fixture' in out['collision'][2]
        assert page.head.status.text.text() == 'in collision'
    finally:
        w.cancel_jobs()
        w.store.undo.setClean()
        w.hide()
        w.deleteLater()
        pump(app, 0.02)
