import pytest

from jarvis_arm.arm.controller import ArmController
from jarvis_arm.config import load_config
from jarvis_arm.drivers.mocks import LinearProjection, MockCamera, MockServoBus, MockToF, MockWorld
from jarvis_arm.perception.grounding import MockGrounder
from jarvis_arm.skills import SkillContext, build_registry
from jarvis_arm.world.scene import Scene


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.servo_bus.default_speed_dps = 2000   # fast mock moves keep tests quick
    c.logging.dir = str(tmp_path / "logs")
    return c


@pytest.fixture
def bus(cfg):
    return MockServoBus([j.id for j in cfg.joints])


@pytest.fixture
def controller(cfg, bus):
    c = ArmController(bus, cfg)
    c.start()
    yield c
    c.shutdown()


@pytest.fixture
def mock_ctx(cfg, controller):
    world = MockWorld()
    proj = LinearProjection(cfg.camera.width, cfg.camera.height)
    said: list[str] = []
    ctx = SkillContext(
        cfg=cfg, controller=controller, camera=MockCamera(cfg.camera.width, cfg.camera.height),
        tof=MockToF(world), grounder=MockGrounder(world, proj), scene=Scene(),
        say=said.append, ask_user=lambda q: None, pixel_to_table=proj.to_table, sim_world=world,
    )
    ctx.said = said
    return ctx


@pytest.fixture
def registry():
    return build_registry()
