"""
stepper_manual.py
==================
Interactive terminal control for stepper + ESC via Nano 33 BLE serial.

Usage:
    python stepper_manual.py            # default COM7
    python stepper_manual.py COM3

Commands (type and press Enter):

  STEPPER
    s <steps>       Step N half-steps  (+CW, -CCW)   e.g.  s 100   s -50
    a <angle>       Step by angle (degrees)           e.g.  a 45    a -90
    g <angle>       Go to absolute angle from home    e.g.  g 180   g 0
    spin [speed]    Spin continuously CW              e.g.  spin 1500
    spinl [speed]   Spin continuously CCW             e.g.  spinl 1500
    stop            Stop spinning / de-energise coils
    spd <us>        Set step delay µs                 e.g.  spd 800
    home            Return to home (0°), ESC to idle

  ESC
    esc <0-100>     Set ESC throttle by percentage    e.g.  esc 50
    esc us <1000-2000>  Set ESC throttle by raw µs    e.g.  esc us 1450
    esc on          Spin up to fire speed (ESC_FIRE_US in firmware)
    esc off         Stop ESC (1000 µs)

  GENERAL
    status          Print current position + ESC throttle
    q               Quit
"""

import sys
import threading
import time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM7'
BAUD = 115200
STEPS_PER_REV = 400    # Sanyo 1.8°/step, half-step mode (200 full-steps × 2)


def connect(port: str) -> serial.Serial:
    print(f"Connecting to {port} at {BAUD} baud...")
    ser = serial.Serial(port, BAUD, timeout=3)
    time.sleep(2.0)                # wait for ESP32 boot
    ser.reset_input_buffer()
    print("Connected.\n")
    return ser


def send(ser: serial.Serial, cmd: str) -> str:
    ser.write((cmd + '\n').encode())
    resp = ser.readline().decode(errors='replace').strip()
    return resp


def deg_to_steps(deg: float) -> int:
    return int(round(deg * STEPS_PER_REV / 360.0))


# ── Continuous spin (runs in background thread) ───────────────────────────────
_spin_active = False
_spin_thread: threading.Thread | None = None


def _spin_loop(ser: serial.Serial, chunk_steps: int, delay_us: int):
    """Send repeated fixed-size chunks until _spin_active goes False."""
    global _spin_active
    send(ser, f"SPEED {delay_us}")
    while _spin_active:
        send(ser, f"PAN {chunk_steps}")


def start_spin(ser: serial.Serial, direction: int, delay_us: int):
    global _spin_active, _spin_thread
    stop_spin(ser)
    _spin_active = True
    chunk = direction * 64           # 64 half-steps per chunk ≈ 11°
    _spin_thread = threading.Thread(
        target=_spin_loop, args=(ser, chunk, delay_us), daemon=True
    )
    _spin_thread.start()
    print(f"Spinning {'CW' if direction > 0 else 'CCW'} at {delay_us} µs/step  (type 'stop' to halt)")


def stop_spin(ser: serial.Serial):
    global _spin_active, _spin_thread
    if _spin_active:
        _spin_active = False
        if _spin_thread:
            _spin_thread.join(timeout=2)
        _spin_thread = None
        send(ser, "STOP")
        print("Stopped.")


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main():
    try:
        ser = connect(PORT)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        print("Check the port name and that the ESP32 is plugged in.")
        sys.exit(1)

    print("Turret manual control ready.  (Note: ESC arming takes ~3s on boot)")
    print("Type 'help' for commands.\n")

    while True:
        try:
            raw = input("turret> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            stop_spin(ser)
            ser.close()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        # ── help ──────────────────────────────────────────────────────────
        if cmd == 'help':
            print("""
  STEPPER
    s <steps>          Step N half-steps (+CW, -CCW)    e.g. s 200  s -100
    a <deg>            Relative angle step               e.g. a 45   a -90
    g <deg>            Go to absolute angle from home    e.g. g 180  g 0
    spin [us/step]     Spin CW continuously              e.g. spin 1500
    spinl [us/step]    Spin CCW continuously
    stop               Stop & de-energise coils
    spd <us>           Set step delay µs (lower=faster)  e.g. spd 800
    home               Return to 0°, ESC to idle

  ESC
    esc <0-100>        Throttle by percentage            e.g. esc 0  esc 60
    esc us <1000-2000> Throttle by raw microseconds      e.g. esc us 1450
    esc on             Spin up to fire speed
    esc off            Stop ESC

  GENERAL
    status             Current pan position + ESC µs
    q / quit           Exit
""")

        # ── step by half-steps ─────────────────────────────────────────────
        elif cmd == 's':
            if len(parts) < 2:
                print("Usage: s <steps>")
                continue
            steps = int(parts[1])
            resp  = send(ser, f"PAN {steps}")
            print(resp)

        # ── step by angle (relative) ───────────────────────────────────────
        elif cmd == 'a':
            if len(parts) < 2:
                print("Usage: a <degrees>")
                continue
            steps = deg_to_steps(float(parts[1]))
            resp  = send(ser, f"PAN {steps}")
            print(f"Stepped {steps} half-steps  →  {resp}")

        # ── go to absolute angle ───────────────────────────────────────────
        elif cmd == 'g':
            if len(parts) < 2:
                print("Usage: g <degrees>")
                continue
            deg  = float(parts[1])
            resp = send(ser, f"PAN_TO {deg:.2f}")
            print(resp)

        # ── continuous spin CW ─────────────────────────────────────────────
        elif cmd == 'spin':
            delay = int(parts[1]) if len(parts) > 1 else 1500
            start_spin(ser, +1, delay)

        # ── continuous spin CCW ────────────────────────────────────────────
        elif cmd == 'spinl':
            delay = int(parts[1]) if len(parts) > 1 else 1500
            start_spin(ser, -1, delay)

        # ── stop ───────────────────────────────────────────────────────────
        elif cmd == 'stop':
            stop_spin(ser)

        # ── set speed ──────────────────────────────────────────────────────
        elif cmd == 'spd':
            if len(parts) < 2:
                print("Usage: spd <microseconds>  (e.g. spd 1000)")
                continue
            us   = int(parts[1])
            resp = send(ser, f"SPEED {us}")
            print(f"Speed set to {us} µs/step  →  {resp}")

        # ── home ───────────────────────────────────────────────────────────
        elif cmd == 'home':
            stop_spin(ser)
            resp = send(ser, "HOME")
            print(resp)

        # ── status ─────────────────────────────────────────────────────────
        elif cmd == 'status':
            resp = send(ser, "STATUS")
            print(resp)

        # ── ESC ────────────────────────────────────────────────────────────
        elif cmd == 'esc':
            if len(parts) < 2:
                print("Usage: esc <0-100>  |  esc us <1000-2000>  |  esc on  |  esc off")
                continue

            sub = parts[1].lower()

            if sub == 'off':
                resp = send(ser, "THROTTLE 1000")
                print(f"ESC stopped  →  {resp}")

            elif sub == 'on':
                resp = send(ser, "THROTTLE 1600")   # matches ESC_FIRE_US in firmware
                print(f"ESC at fire speed  →  {resp}")

            elif sub == 'us':
                if len(parts) < 3:
                    print("Usage: esc us <1000-2000>")
                    continue
                us   = int(parts[2])
                us   = max(1000, min(2000, us))
                resp = send(ser, f"THROTTLE {us}")
                print(f"ESC {us} µs  →  {resp}")

            else:
                # percentage 0–100
                try:
                    pct = float(sub)
                except ValueError:
                    print("Usage: esc <0-100>")
                    continue
                pct  = max(0.0, min(100.0, pct))
                us   = int(1000 + pct * 10)   # 0%→1000µs, 100%→2000µs
                resp = send(ser, f"THROTTLE {us}")
                print(f"ESC {pct:.0f}%  ({us} µs)  →  {resp}")

        # ── quit ───────────────────────────────────────────────────────────
        elif cmd in ('q', 'quit'):
            stop_spin(ser)
            send(ser, "THROTTLE 1000")   # always idle ESC before exit
            ser.close()
            print("Bye.")
            break

        else:
            print(f"Unknown command '{cmd}'.  Type 'help' for list.")


if __name__ == "__main__":
    main()
