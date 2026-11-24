"""Feetech STS3215 serial bus.

Everything here is in RAW servo units (ticks 0..4095, 1 tick = 360/4096 deg,
speed in ticks/s). Joint-angle conversion lives in arm/kinematics.JointMap.

Only ONE thread may touch a ServoBus: the bus-owner thread in arm/controller.py.

FeetechServoBus is written against the STS3215 control table and the
`feetech-servo-sdk` (scservo_sdk) package but is UNTESTED ON HARDWARE.
Bring it up with tools/ scripts one servo at a time before trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

TICKS_PER_REV = 4096
TICK_MAX = TICKS_PER_REV - 1

# STS3215 control table (SMS/STS series, protocol end 0)
ADDR_ID = 5
ADDR_TORQUE_ENABLE = 40
ADDR_ACCEL = 41
ADDR_GOAL_POSITION = 42      # 2B pos, 2B time, 2B speed  -> 6 bytes from 42
ADDR_LOCK = 55
ADDR_PRESENT_POSITION = 56   # 2B pos, 2B speed, 2B load, 1B volt, 1B temp -> 8 bytes from 56


@dataclass(frozen=True)
class ServoTelemetry:
    position: int        # ticks
    load_pct: float      # signed, % of max torque
    voltage_v: float
    temp_c: float


class ServoBusError(RuntimeError):
    pass


class ServoBus(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def ping(self, servo_id: int) -> bool: ...
    def set_torque(self, ids: Iterable[int], enable: bool) -> None: ...
    def sync_write_goals(self, goals: dict[int, int], speed: int) -> None: ...
    def read_positions(self, ids: Iterable[int]) -> dict[int, int]: ...
    def read_telemetry(self, ids: Iterable[int]) -> dict[int, ServoTelemetry]: ...


def _decode_signed(raw: int, sign_bit: int) -> int:
    """Feetech uses sign-magnitude encoding for speed/load."""
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if raw & (1 << sign_bit) else mag


class FeetechServoBus:
    """Real bus via Waveshare Bus Servo Adapter (A). UNTESTED ON HARDWARE."""

    def __init__(self, port: str, baudrate: int = 1_000_000):
        self.port_name = port
        self.baudrate = baudrate
        self._port = None
        self._pkt = None

    def open(self) -> None:
        try:
            import scservo_sdk as scs  # type: ignore
        except ImportError as e:
            raise ServoBusError("feetech-servo-sdk not installed: pip install -e '.[hw]'") from e
        self._scs = scs
        self._port = scs.PortHandler(self.port_name)
        self._pkt = scs.PacketHandler(0)  # STS = protocol end 0
        if not self._port.openPort():
            raise ServoBusError(f"cannot open {self.port_name}")
        if not self._port.setBaudRate(self.baudrate):
            raise ServoBusError(f"cannot set baudrate {self.baudrate}")

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
            self._port = None

    def _check(self, comm: int, err: int, what: str) -> None:
        if comm != self._scs.COMM_SUCCESS:
            raise ServoBusError(f"{what}: {self._pkt.getTxRxResult(comm)}")
        if err:
            raise ServoBusError(f"{what}: servo error {self._pkt.getRxPacketError(err)}")

    def ping(self, servo_id: int) -> bool:
        _, comm, err = self._pkt.ping(self._port, servo_id)
        return comm == self._scs.COMM_SUCCESS and not err

    def set_torque(self, ids: Iterable[int], enable: bool) -> None:
        for i in ids:
            comm, err = self._pkt.write1ByteTxRx(self._port, i, ADDR_TORQUE_ENABLE, 1 if enable else 0)
            self._check(comm, err, f"torque id={i}")

    def sync_write_goals(self, goals: dict[int, int], speed: int) -> None:
        scs = self._scs
        gsw = scs.GroupSyncWrite(self._port, self._pkt, ADDR_GOAL_POSITION, 6)
        spd = max(0, min(int(speed), 0x7FFF))
        for i, pos in goals.items():
            pos = max(0, min(int(pos), TICK_MAX))
            data = [pos & 0xFF, pos >> 8, 0, 0, spd & 0xFF, spd >> 8]
            if not gsw.addParam(i, data):
                raise ServoBusError(f"sync write addParam failed id={i}")
        comm = gsw.txPacket()
        gsw.clearParam()
        if comm != scs.COMM_SUCCESS:
            raise ServoBusError(f"sync write: {self._pkt.getTxRxResult(comm)}")

    def _read_block(self, i: int) -> list[int]:
        data, comm, err = self._pkt.readTxRx(self._port, i, ADDR_PRESENT_POSITION, 8)
        self._check(comm, err, f"read id={i}")
        return data

    def read_positions(self, ids: Iterable[int]) -> dict[int, int]:
        out = {}
        for i in ids:
            pos, comm, err = self._pkt.read2ByteTxRx(self._port, i, ADDR_PRESENT_POSITION)
            self._check(comm, err, f"read pos id={i}")
            out[i] = pos
        return out

    def read_telemetry(self, ids: Iterable[int]) -> dict[int, ServoTelemetry]:
        out = {}
        for i in ids:
            d = self._read_block(i)
            pos = d[0] | (d[1] << 8)
            load_raw = d[4] | (d[5] << 8)
            out[i] = ServoTelemetry(
                position=pos,
                load_pct=_decode_signed(load_raw, 10) / 10.0,
                voltage_v=d[6] / 10.0,
                temp_c=float(d[7]),
            )
        return out
