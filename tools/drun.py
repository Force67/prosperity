#!/usr/bin/env python3
"""drun.py: one debugging round against a title, in one command.

Runs ps4delta headless inside the nix dev shell, keeps a filtered log, and when
the title stops presenting frames (or at the times asked for) probes every
thread with SIGUSR1 and prints each one's guest and host stack, symbolized.
Ends with a compact report: the fps timeline, asserts, crashes, warnings and
the probed stacks, grouped.

  tools/drun.py uncharted2 -t 90                 # run 90 s, probe on a stall
  tools/drun.py uc2 -t 120 -o DELTA_GPU_CSSYNC=1 -g cstime
  tools/drun.py uc2 --pad "12:cross,20:cross"    # scripted pad presses
  tools/drun.py uc2 --probe-at 50,80 --film 400  # probes + a filmstrip
  tools/drun.py --replay /tmp/drun/run.log       # re-report a saved run
  tools/drun.py uc2 --peek big2+0x10508d0,0x1200100000   # u64s at each probe

A title is a pkg path or a substring of a file in ~/Documents/dumps.
"""
import argparse
import collections
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(ROOT, "build", "delta", "main")
BIN = os.path.join(BIN_DIR, "ps4delta")
DUMPS = os.path.expanduser("~/Documents/dumps")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"^\[\s*([0-9.]+)\]")
MODFRAME = re.compile(r"\b([0-9a-f]{12,16})\s+([A-Za-z][\w-]*)\+0x([0-9a-f]+) \(")
ALIASES = {"uc2": "uncharted2", "isaac": "Isaac", "gta": "gtasa",
           "tr": "tr.pkg", "pt": "PT.pkg", "ut": "Undertale", "doom": "doom64"}


def find_pkg(name):
    if os.path.exists(name):
        return name
    key = ALIASES.get(name, name).lower()
    for f in sorted(os.listdir(DUMPS)):
        if key in f.lower() and f.endswith((".pkg", ".ffpkg", ".rar", ".zip")):
            return os.path.join(DUMPS, f)
    sys.exit(f"no title matching {name!r} in {DUMPS}")


def in_nix():
    return bool(os.environ.get("IN_NIX_SHELL"))


def stamp(line):
    m = STAMP.match(line)
    return float(m.group(1)) if m else None


class Run:
    def __init__(self, args):
        self.a = args
        self.fps = []            # (t, fps)
        self.io = []             # (t, bytes read so far)
        self.asserts = collections.Counter()
        self.crash = []          # crash handler lines
        self.crashed = False
        self.warnings = collections.Counter()
        self.greps = []
        self.tail = collections.deque(maxlen=args.grep_max // 2)
        self.probes = []         # (label, lines)
        self.cur_probe = None
        self.last_fps_t = None   # wall clock of the last fps line
        self.base = None         # ps4delta load base, for addr2line
        self.started = time.time()
        self.log = None
        self.modules = {}        # guest module name -> load base

    # --- log -------------------------------------------------------------
    def feed_log(self, line):
        if self.log:
            self.log.write(line + "\n")

    def replay_marker(self, line):
        m = re.match(r"#drun base=(0x[0-9a-f]+)", line)
        if m:
            self.base = int(m.group(1), 16)
        m = re.match(r"#drun probe (.*)", line)
        if m:
            self.cur_probe = []
            self.probes.append((m.group(1), self.cur_probe))

    def feed(self, line):
        a = self.a
        # Any symbolized address ("<hex> module+0xoff") names its module base.
        m = MODFRAME.search(line)
        if m and m.group(2) not in self.modules:
            self.modules[m.group(2)] = int(m.group(1), 16) - int(m.group(3), 16)
        if line.startswith("#drun"):
            self.replay_marker(line)
            return
        if "[fps]" in line:
            m = re.search(r"\[fps\] ([0-9.]+) fps", line)
            t = stamp(line)
            if m and t is not None:
                self.fps.append((t, float(m.group(1))))
                self.io.append((t, self.read_bytes()))
            self.last_fps_t = time.time()
        if "[assert]" in line:
            m = re.search(r"@ (\S+)", line)
            self.asserts[m.group(1) if m else line[:80]] += 1
        if "entered sig=" in line:
            self.crashed = True
        if self.crashed and "crashHandler" in line and "maps" not in line:
            if len(self.crash) < 60:
                self.crash.append(line)
        if "<Warning>" in line or "<Error>" in line:
            key = re.sub(r"0x[0-9a-f]+|\b\d+\b", "#", line.split("> ", 1)[-1])
            self.warnings[key[:140]] += 1
        if a.grep and any(g in line for g in a.grep):
            self.keep(line)
        if "[probe]" in line or "[waitprobe]" in line:
            if self.cur_probe is not None:
                self.cur_probe.append(line)
        if "[waitwatch]" in line or "[wprot]" in line:
            self.keep(line)

    def keep(self, line):
        # The first and the latest matches: a run's story is how it started
        # and where it ended up.
        if len(self.greps) < self.a.grep_max // 2:
            self.greps.append(line)
        else:
            self.tail.append(line)

    def read_bytes(self):
        # What the emulator has read from its files so far (rchar counts
        # the pkg reads, cached or not).
        pid = self.child_pid(None) if self.log else None
        try:
            with open(f"/proc/{pid}/io") as f:
                for l in f:
                    if l.startswith("rchar:"):
                        return int(l.split()[1])
        except (OSError, TypeError):
            pass
        return None

    # --- probing ---------------------------------------------------------
    def probe(self, proc, label):
        pid = self.child_pid(proc)
        if not pid:
            return
        if self.base is None:
            try:
                with open(f"/proc/{pid}/maps") as f:
                    for l in f:
                        if l.rstrip().endswith("/ps4delta"):
                            self.base = int(l.split("-")[0], 16)
                            self.feed_log(f"#drun base={self.base:#x}")
                            break
            except OSError:
                pass
        self.feed_log(f"#drun probe {label}")
        lines = []
        self.probes.append((label, lines))
        self.cur_probe = lines
        for spec in filter(None, self.a.peek.split(",")):
            addr = self.resolve(spec)
            try:
                with open(f"/proc/{pid}/mem", "rb") as f:
                    f.seek(addr)
                    v = int.from_bytes(f.read(8), "little")
                lines.append(f"[peek] {spec} ({addr:#x}) = {v:#x} ({v})")
            except (OSError, ValueError, TypeError) as e:
                lines.append(f"[peek] {spec}: {e}")
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            return
        for tid in tids:
            try:
                os.kill(int(tid), signal.SIGUSR1)
            except OSError:
                pass
            time.sleep(0.01)

    def resolve(self, spec):
        # module+0xoff against the module bases the log reported.
        if "+" not in spec:
            return int(spec, 16)
        mod, off = spec.split("+", 1)
        base = self.modules.get(mod)
        if base is None:
            for name, b in self.modules.items():
                if name.startswith(mod):
                    base = b
                    break
        return None if base is None else base + int(off, 16)

    @staticmethod
    def child_pid(proc):
        # The emulator runs under `nix develop`; find it among descendants.
        try:
            out = subprocess.run(["pgrep", "-x", "ps4delta"], capture_output=True,
                                 text=True).stdout.split()
        except OSError:
            return None
        return int(out[0]) if out else None

    # --- report ----------------------------------------------------------
    def symbolize(self, addrs):
        # Only addresses inside the binary: a guest frame below it would make
        # a negative offset that addr2line parses as an option.
        addrs = [x for x in addrs if self.base is not None and
                 self.base <= x < self.base + (1 << 32)]
        if not addrs:
            return {}
        offs = [hex(x - self.base) for x in addrs]
        out = subprocess.run(["addr2line", "-f", "-C", "-e", BIN] + offs,
                             capture_output=True, text=True).stdout.splitlines()
        names = {}
        for i, a in enumerate(addrs):
            fn = out[2 * i] if 2 * i < len(out) else "?"
            fn = re.sub(r"\(.*", "", fn)  # drop parameter lists
            names[a] = fn if fn != "??" else f"{a:#x}"
        return names

    def report_probes(self):
        for label, lines in self.probes:
            threads = []  # (tid, gtid, rip, guest frames, host addrs)
            cur = None
            for l in lines:
                m = re.search(r"\[probe\] tid=(\d+) gtid=(\d+) rip=\S+ (.*)", l)
                if m:
                    cur = [m.group(1), m.group(2), m.group(3).strip(), [], []]
                    threads.append(cur)
                    continue
                if cur is None:
                    continue
                m = re.search(r"sp\+\S+\s+\S+\s+(\S+\+0x[0-9a-f]+)", l)
                if m and len(cur[3]) < self.a.depth:
                    cur[3].append(m.group(1))
                m = re.search(r"hostbt:(.*)", l)
                if m:
                    cur[4] = [int(x, 16) for x in m.group(1).split()]
            host = sorted({a for t in threads for a in t[4]})
            names = self.symbolize(host)
            skip = ("ProbeHandler", "__restore_rt", "backtrace", "killpg",
                    "Probe", "SignalHandler")
            groups = collections.defaultdict(list)
            for tid, gtid, rip, guest, hb in threads:
                hs = [names.get(a, hex(a)) for a in hb]
                hs = [h for h in hs if not any(s in h for s in skip)]
                hs = [h for h in hs if "?" not in h and not h.startswith("0x")]
                hs = hs[: self.a.depth]
                key = (" < ".join(guest) or rip, " < ".join(hs))
                groups[key].append(f"{tid}/g{gtid}")
            print(f"\n== probe {label}: {len(threads)} threads")
            for (g, h), who in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                if self.a.idle_filter and not g.strip("?() ") and (
                        "Sleep" in h or "Wait" in h) and "Walk" not in h:
                    continue
                print(f"  x{len(who)} [{','.join(who[:6])}{'...' if len(who) > 6 else ''}]")
                print(f"     guest: {g}")
                if h:
                    print(f"     host:  {h}")
            for l in lines:
                if l.startswith("[peek]"):
                    print("  " + l)
            waits = [l for l in lines if "[waitprobe]" in l]
            if waits:
                print("  parked:")
                for w in waits[-12:]:
                    print("   ", w.split("[waitprobe]")[-1].strip())

    def report(self, exit_code, log_path):
        a = self.a
        print(f"\n=== {os.path.basename(a.pkg or a.replay)}  ran {time.time() - self.started:.0f}s  "
              f"exit={exit_code}  log={log_path}")
        if self.fps:
            tl = " ".join(f"{t:.0f}:{f:.0f}" for t, f in self.fps)
            print(f"fps  {tl}")
            last = self.fps[-1][0]
            print(f"last frame report at t={last:.0f}s")
            rates = []
            for (t0, b0), (t1, b1) in zip(self.io, self.io[1:]):
                if b0 is not None and b1 is not None and t1 > t0:
                    rates.append(f"{t1:.0f}:{(b1 - b0) / (t1 - t0) / 1e6:.1f}")
            if rates:
                print("io MB/s " + " ".join(rates))
        else:
            print("fps  (never presented)")
        if self.asserts:
            print("asserts " + ", ".join(f"{k} x{v}" for k, v in self.asserts.most_common(8)))
        if self.crash:
            print("CRASH:")
            text = "\n".join(self.crash)
            rip = re.search(r"\brip\s+=\s+([0-9a-f]+)", text)
            host = re.search(r"host rip = \S*ps4delta\+0x([0-9a-f]+)", text)
            if rip and host and self.base is None:
                self.base = int(rip.group(1), 16) - int(host.group(1), 16)
            addrs = [int(x, 16) for x in
                     re.findall(r"(?:rip\s+=|#\d+)\s+([0-9a-f]{12,16})", text)]
            names = self.symbolize([x for x in addrs if x >> 40 == 0x55])
            for l in self.crash[:14]:
                l = l.split("] ", 1)[-1]
                for x, n in names.items():
                    l = l.replace(f"{x:016x}", f"{x:016x} {n}")
                print("  " + l[:200])
        if self.warnings:
            print("warnings:")
            for k, v in self.warnings.most_common(a.warn_max):
                print(f"  x{v} {k}")
        if self.greps:
            print("grep:")
            for l in self.greps:
                print("  " + l[:220])
            if self.tail:
                print("  ...")
                for l in self.tail:
                    print("  " + l[:220])
        self.report_probes()


def film(dir_, out):
    frames = sorted(f for f in os.listdir(dir_) if f.startswith("every_"))
    if not frames:
        return None
    cols = 4
    inputs, filt = [], []
    for i, f in enumerate(frames[:16]):
        inputs += ["-i", os.path.join(dir_, f)]
        filt.append(f"[{i}]scale=480:-1[s{i}]")
    n = min(len(frames), 16)
    rows = (n + cols - 1) // cols
    layout = "|".join(f"{(i % cols) * 480}_{(i // cols) * 270}" for i in range(n))
    chain = "".join(f"[s{i}]" for i in range(n))
    fc = ";".join(filt) + f";{chain}xstack=inputs={n}:layout={layout}:fill=black" \
        if n > 1 else f"{filt[0]};[s0]null"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + inputs +
                   ["-filter_complex", fc, out])
    return out if os.path.exists(out) else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pkg", nargs="?")
    p.add_argument("--replay", help="re-report a saved run.log")
    p.add_argument("--peek", default="",
                   help="comma list of addresses (hex, or module+0xoff) whose "
                        "u64 is read at every probe")
    p.add_argument("-t", "--time", type=float, default=90, help="seconds to run")
    p.add_argument("-o", "--opt", action="append", default=[],
                   help="Name=Value option (repeatable)")
    p.add_argument("-g", "--grep", action="append", default=[],
                   help="keep lines containing this (repeatable)")
    p.add_argument("--grep-max", type=int, default=60)
    p.add_argument("--warn-max", type=int, default=10)
    p.add_argument("--stall", type=float, default=10,
                   help="probe when no frame for this many seconds (0 = off)")
    p.add_argument("--probe-at", default="", help="comma list of seconds")
    p.add_argument("--pad", default="", help="DELTA_PAD_SCRIPT, seconds:button")
    p.add_argument("--film", type=int, default=0,
                   help="snapshot every N frames into a filmstrip")
    p.add_argument("--window", action="store_true", help="show a real window")
    p.add_argument("--depth", type=int, default=7, help="frames per stack")
    p.add_argument("--no-idle-filter", dest="idle_filter", action="store_false",
                   help="also print threads idling in host waits")
    p.add_argument("--keep-going", action="store_true",
                   help="do not stop the run after a stall probe")
    p.add_argument("--out", default=os.environ.get("DRUN_OUT", "/tmp/drun"))
    a = p.parse_args()
    if a.replay:
        run = Run(a)
        with open(a.replay) as f:
            for line in f:
                run.feed(line.rstrip("\n"))
        run.report(None, a.replay)
        return
    if not a.pkg:
        p.error("a title is required")

    if not in_nix():
        os.execvp("nix", ["nix", "develop", ROOT, "--command", sys.executable]
                  + sys.argv)

    a.pkg = find_pkg(a.pkg)
    os.makedirs(a.out, exist_ok=True)
    film_dir = os.path.join(a.out, "film")
    shutil.rmtree(film_dir, ignore_errors=True)
    opts = list(a.opt)
    if a.pad:
        opts.append(f"DELTA_PAD_SCRIPT={a.pad}")
    if a.film:
        os.makedirs(film_dir)
        opts += [f"DELTA_GPU_SNAPEVERY={a.film}", "DELTA_GPU_SNAPEVERY_MAX=16",
                 f"DELTA_GPU_DUMP_DIR={film_dir}"]
    env = dict(os.environ)
    if not a.window:
        env["SDL_VIDEODRIVER"] = "offscreen"
    subprocess.run(["pkill", "-9", "-x", "ps4delta"])
    cmd = [BIN, a.pkg] + [f"+{o}" for o in opts]
    log_path = os.path.join(a.out, "run.log")
    log = open(log_path, "w")
    log_budget = 300 << 20
    proc = subprocess.Popen(cmd, cwd=BIN_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    run = Run(a)
    run.log = log
    probe_at = sorted(float(x) for x in a.probe_at.split(",") if x)
    stall_probes = 0
    buf = b""
    try:
        while True:
            now = time.time() - run.started
            if now > a.time:
                break
            if probe_at and now >= probe_at[0]:
                run.probe(proc, f"t={probe_at.pop(0):.0f}s")
            if (a.stall and run.last_fps_t and
                    time.time() - run.last_fps_t > a.stall and stall_probes < 2):
                stall_probes += 1
                run.probe(proc, f"stall #{stall_probes} at t={now:.0f}s")
                run.last_fps_t = time.time() - a.stall + 2  # next probe in 2 s
                if stall_probes == 2 and not a.keep_going:
                    time.sleep(1.5)
                    a.time = now + 2
            r, _, _ = select.select([proc.stdout], [], [], 0.2)
            if r:
                chunk = os.read(proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    line = ANSI.sub("", raw.decode(errors="replace"))
                    if log_budget > 0:
                        log.write(line + "\n")
                        log_budget -= len(line)
                    run.feed(line)
            elif proc.poll() is not None:
                break
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        subprocess.run(["pkill", "-9", "-x", "ps4delta"])
        log.close()
    run.report(proc.poll(), log_path)
    if a.film:
        out = film(film_dir, os.path.join(a.out, "film.png"))
        print(f"\nfilm: {out}" if out else "\nfilm: (no frames)")


if __name__ == "__main__":
    main()
