"""ArmController: the ONLY owner of the servo bus.

All bus I/O happens on one thread (`_run`). Other threads submit commands via
a queue and read cached state. halt() is safe to call from any thread and
takes effect on the next control tick (default 20 ms): every servo's goal is
set to its present position with torque on (torque-hold).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable

from ..config import Config
from ..drivers.servo_bus import ServoBus, ServoTelemetry
from .kinematics import JointMap
from .safety import SafetyViolation, check_joint_targets

log = logging.getLogger(__name__)

POSITION_TOL_DEG = 2.0


@dataclass
class MoveResult:
    ok: bool
    reason: str = ""


class ArmController:
    def __init__(self, bus: ServoBus, cfg: Config):
        self.bus = bus
        self.cfg = cfg
        self.jm = JointMap(cfg.joints)
        self._joints = {j.name: j for j in cfg.joints}
        self._cmds: queue.Queue[tuple[Callable[[ServoBus], object], Future]] = queue.Queue()
        self._halt_req = threading.Event()
        self._halted = threading.Event()
        self._halt_reason = ""
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._pos_deg: dict[str, float] = {}
        self._telemetry: dict[str, ServoTelemetry] = {}
        self._last_tick = 0.0
        self._overload_since: dict[str, float] = {}
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.bus.open()
        missing = [j.name for j in self.cfg.joints if not self.bus.ping(j.id)]
        if missing:
            raise RuntimeError(f"servos not responding: {missing}")
        self._update_positions(self.bus.read_positions(self.jm.ids))
        self.bus.set_torque(self.jm.ids, True)
        self._last_tick = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="bus-owner", daemon=True)
        self._thread.start()
        log.info("arm controller started")

    def shutdown(self, release_torque: bool = False) -> None:
        self.halt("shutdown")
        time.sleep(2.0 / self.cfg.servo_bus.control_hz)
        if release_torque:
            self._submit(lambda b: b.set_torque(self.jm.ids, False)).result(timeout=1)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.bus.close()

    # ------------------------------------------------------------- state
    def _update_positions(self, ticks: dict[int, int]) -> None:
        with self._state_lock:
            for j in self.cfg.joints:
                if j.id in ticks:
                    self._pos_deg[j.name] = self.jm.ticks_to_deg(j.name, ticks[j.id])

    def joint_positions(self) -> dict[str, float]:
        with self._state_lock:
            return dict(self._pos_deg)

    def telemetry(self) -> dict[str, ServoTelemetry]:
        with self._state_lock:
            return dict(self._telemetry)

    def is_alive(self, max_stall_s: float = 0.5) -> bool:
        """Watchdog: bus thread ticked recently."""
        return self._thread is not None and self._thread.is_alive() and \
            time.monotonic() - self._last_tick < max_stall_s

    @property
    def halted(self) -> bool:
        return self._halted.is_set()

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    # -------------------------------------------------------------- halt
    def halt(self, reason: str = "halt") -> None:
        if not self._halted.is_set():
            log.log(logging.INFO if reason == "shutdown" else logging.WARNING, "HALT: %s", reason)
        self._halt_reason = reason
        self._halted.set()
        self._halt_req.set()

    def resume(self) -> None:
        # Let a pending hold land first, or it would swallow the next command.
        deadline = time.monotonic() + 1.0
        while self._halt_req.is_set() and self.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        self._halted.clear()
        self._halt_reason = ""
        log.info("halt cleared")

    # -------------------------------------------------------------- motion
    def move_joints(self, targets_deg: dict[str, float], speed_dps: float | None = None,
                    wait: bool = True, timeout_s: float = 15.0) -> MoveResult:
        if self.halted:
            return MoveResult(False, f"arm halted ({self._halt_reason}); say 'resume' first")
        try:
            check_joint_targets(targets_deg, self._joints)
        except SafetyViolation as e:
            return MoveResult(False, f"refused: {e}")
        speed = self.jm.dps_to_tick_speed(speed_dps or self.cfg.servo_bus.default_speed_dps)
        goals = {self._joints[n].id: self.jm.deg_to_ticks(n, d) for n, d in targets_deg.items()}
        try:
            self._submit(lambda b: b.sync_write_goals(goals, speed)).result(timeout=1)
        except Exception as e:  # noqa: BLE001
            if self.halted:
                return MoveResult(False, f"halted ({self._halt_reason})")
            return MoveResult(False, f"bus error: {e}")
        if not wait:
            return MoveResult(True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.halted:
                return MoveResult(False, f"halted during move ({self._halt_reason})")
            pos = self.joint_positions()
            if all(abs(pos[n] - d) <= POSITION_TOL_DEG for n, d in targets_deg.items()):
                return MoveResult(True)
            time.sleep(0.02)
        return MoveResult(False, "move timed out (blocked or stalled?)")

    def move_to_named(self, pose: str, **kw) -> MoveResult:
        vals = self.cfg.named_poses.get(pose)
        if vals is None:
            return MoveResult(False, f"unknown pose '{pose}'; known: {sorted(self.cfg.named_poses)}")
        return self.move_joints(dict(zip([j.name for j in self.cfg.arm_joints], vals)), **kw)

    def set_gripper(self, deg: float, **kw) -> MoveResult:
        return self.move_joints({"gripper": deg}, **kw)

    # ---------------------------------------------------------- bus thread
    def _submit(self, fn: Callable[[ServoBus], object]) -> Future:
        fut: Future = Future()
        if self._thread is None or not self._thread.is_alive():
            fut.set_exception(RuntimeError("controller not running"))
        else:
            self._cmds.put((fn, fut))
        return fut

    def _run(self) -> None:
        period = 1.0 / self.cfg.servo_bus.control_hz
        tel_period = 1.0 / self.cfg.servo_bus.telemetry_hz
        next_tel = 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self._halt_req.is_set():
                    self._do_hold()
                else:
                    self._drain_commands()
                if t0 >= next_tel:
                    self._read_telemetry()
                    next_tel = t0 + tel_period
                else:
                    self._update_positions(self.bus.read_positions(self.jm.ids))
            except Exception as e:  # noqa: BLE001
                log.exception("bus thread error")
                self.halt(f"bus error: {e}")
            self._last_tick = time.monotonic()
            time.sleep(max(0.0, period - (self._last_tick - t0)))

    def _drain_commands(self) -> None:
        while True:
            try:
                fn, fut = self._cmds.get_nowait()
            except queue.Empty:
                return
            if self._halt_req.is_set():
                fut.set_exception(RuntimeError("halted"))
                continue
            try:
                fut.set_result(fn(self.bus))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def _do_hold(self) -> None:
        self._halt_req.clear()
        # Drop anything queued (it was issued before the halt).
        while True:
            try:
                _, fut = self._cmds.get_nowait()
                fut.set_exception(RuntimeError("halted"))
            except queue.Empty:
                break
        present = self.bus.read_positions(self.jm.ids)
        self.bus.set_torque(self.jm.ids, True)
        self.bus.sync_write_goals(present, speed=0)
        self._update_positions(present)

    def _read_telemetry(self) -> None:
        tel = self.bus.read_telemetry(self.jm.ids)
        by_name = {j.name: tel[j.id] for j in self.cfg.joints if j.id in tel}
        with self._state_lock:
            self._telemetry = by_name
        self._update_positions({i: t.position for i, t in tel.items()})
        sb = self.cfg.servo_bus
        now = time.monotonic()
        for name, t in by_name.items():
            # Sustained overload only: brief spikes while accelerating are normal.
            if abs(t.load_pct) > sb.max_load_pct and name != "gripper":
                since = self._overload_since.setdefault(name, now)
                if now - since >= sb.overload_hold_s:
                    self.halt(f"{name} overload {t.load_pct:.0f}% for {now - since:.1f}s")
            else:
                self._overload_since.pop(name, None)
            if t.temp_c > sb.max_temp_c:
                self.halt(f"{name} overtemp {t.temp_c:.0f}C")
