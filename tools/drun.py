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
  tools/drun.py uc2 -t 200 --capture t=100      # GPU frame capture + summary
  tools/drun.py uc2 -t 130 --tracy 100:20        # Tracy profile of 100..120 s
  tools/drun.py uc2 -t 130 --perf 100:10:NdJob   # perf samples of matching threads
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
        self.modmap = []         # (base, size, name) from [modmap] lines
        self.capture_started = False
        self.capture_done = False

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
        m = re.search(r"\[modmap\] (\S+) base=(0x[0-9a-f]+) size=(0x[0-9a-f]+)"
                      r"(?: ehhdr=(0x[0-9a-f]+))?", line)
        if m:
            self.modmap.append((int(m.group(2), 16), int(m.group(3), 16),
                                m.group(1), int(m.group(4) or "0", 16)))
        if "[gpucap] capturing frame" in line:
            self.capture_started = True
        if "[gpucap] capture complete" in line:
            self.capture_done = True
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


def thread_times(pid):
    """{tid: (name, cpu s, kernel s, minor faults)} from /proc, guest code
    included."""
    out = {}
    tick = os.sysconf("SC_CLK_TCK")
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        name = stat[stat.index("(") + 1:stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        out[int(tid)] = (name, (int(fields[11]) + int(fields[12])) / tick,
                         int(fields[12]) / tick, int(fields[7]))
    return out


# Zones that are a thread parked, not a thread working.
WAIT_ZONE = re.compile(r"sleep|wait|umtx|kevent|yield|select|poll|drain|"
                       r"nanosleep|sigsuspend|evf|sema|cond", re.I)


class Tracy:
    """tracy-capture against the on-demand client, from START for SECS."""

    def __init__(self, spec, out):
        start, _, secs = spec.partition(":")
        self.start = float(start)
        self.secs = float(secs) if secs else 0
        self.path = os.path.join(out, "trace.tracy")
        self.proc = None
        self.t0 = 0
        self.threads_start = {}
        self.threads_end = {}
        if os.path.exists(self.path):
            os.remove(self.path)

    def poll(self, now, pid):
        if self.proc:
            if self.secs and not self.threads_end and \
                    now >= self.start + self.secs:
                self.threads_end = thread_times(pid)
                self.t1 = time.time()
            return
        if now < self.start:
            return
        cmd = ["tracy-capture", "-o", self.path, "-a", "127.0.0.1", "-f"]
        if self.secs:
            cmd += ["-s", str(int(self.secs))]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self.threads_start = thread_times(pid)
        self.t0 = time.time()
        self.t1 = 0

    def report(self, top):
        if not self.proc:
            print("\ntracy: the run ended before the capture started")
            return
        try:
            self.proc.wait(timeout=120)  # it saves once the client is gone
        except subprocess.TimeoutExpired:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=60)
        if not os.path.exists(self.path):
            print("\ntracy: no trace written")
            return
        self_rows = self.cpu_zones(["-e"])  # self time: where the work is
        incl = {r["name"]: r for r in self.cpu_zones([])}
        frames = incl.get("gpu.end_frame", {}).get("counts", 0)
        print(f"\n== tracy {self.path}  "
              f"({os.path.getsize(self.path) / 1e6:.0f} MB, {frames} frames)")
        self.report_threads(frames)
        work = [r for r in self_rows if not WAIT_ZONE.search(r["name"])]
        waits = [r for r in self_rows if WAIT_ZONE.search(r["name"])]
        self.print_zones("cpu work (self time)", work, incl, frames, top)
        self.print_zones("waits (self time: parked, not working)", waits, incl,
                         frames, 12)
        self.report_gpu(frames, top)
        print("  open it: tracy " + self.path)

    def report_threads(self, frames):
        end = self.threads_end
        if not end or not self.threads_start:
            return
        wall = (self.t1 or time.time()) - self.t0
        rows = []
        for tid, (name, cpu, sys_s, faults) in end.items():
            before = self.threads_start.get(tid, (name, 0, 0, 0))
            if cpu - before[1] > 0:
                rows.append((cpu - before[1], tid, name, sys_s - before[2],
                             faults - before[3]))
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows)
        print(f"  threads by CPU over {wall:.0f} s (all threads {total / wall:.2f} "
              f"cores; guest code shows here, not in zones):")
        for cpu, tid, name, sys_s, faults in rows[:12]:
            per_frame = (f"  {cpu * 1e3 / frames:7.2f} ms/frame "
                         f"{faults / frames:6.0f} faults/frame" if frames else "")
            print(f"    {name[:16]:16} tid {tid:<8} {cpu / wall * 100:5.1f}% "
                  f"(kernel {sys_s / max(cpu, 1e-9) * 100:3.0f}%){per_frame}")

    def print_zones(self, title, rows, incl, frames, top):
        if not rows:
            return
        print(f"  {title}:")
        print(f"    {'zone':34} {'self ms':>9} {'incl ms':>9} {'calls':>9} "
              f"{'us/call':>9}" + ("  ms/frame" if frames else ""))
        rows.sort(key=lambda r: -r["total_ns"])
        for r in rows[:top]:
            inc = incl.get(r["name"], r)["total_ns"]
            per_frame = (f"  {r['total_ns'] / frames / 1e6:8.2f}"
                         if frames else "")
            print(f"    {r['name'][:34]:34} {r['total_ns'] / 1e6:9.1f} "
                  f"{inc / 1e6:9.1f} {r['counts']:9d} "
                  f"{r['total_ns'] / max(r['counts'], 1) / 1e3:9.1f}"
                  + per_frame)

    def report_gpu(self, frames, top):
        total = collections.Counter()
        count = collections.Counter()
        for v in self.csv(["-g"]):
            if len(v) >= 4 and v[3].isdigit():
                total[v[0]] += int(v[3])
                count[v[0]] += 1
        if not total:
            return
        busy = sum(total.values())
        print(f"  gpu (timestamped regions and dispatches, "
              f"{busy / max(frames, 1) / 1e6:.2f} ms/frame busy):")
        for name, ns in total.most_common(min(top, 15)):
            per_frame = f"  {ns / frames / 1e6:8.2f} ms/frame" if frames else ""
            print(f"    {name[:34]:34} {ns / 1e6:9.1f} ms {count[name]:7d}x"
                  + per_frame)

    def csv(self, flags):
        out = subprocess.run(["tracy-csvexport"] + flags + [self.path],
                             capture_output=True, text=True).stdout
        return [l.split(",") for l in out.splitlines()]

    def cpu_zones(self, flags):
        lines = self.csv(flags)
        if not lines:
            return []
        head = lines[0]
        rows = []
        for v in lines[1:]:
            if len(v) != len(head):
                continue
            r = dict(zip(head, v))
            try:
                rows.append({"name": r["name"],
                             "total_ns": int(r["total_ns"]),
                             "counts": int(r["counts"])})
            except (KeyError, ValueError):
                pass
        return rows


class Perf:
    """perf record of chosen threads, reported by guest module+offset."""

    def __init__(self, spec, out):
        parts = spec.split(":")
        self.start = float(parts[0])
        self.secs = float(parts[1]) if len(parts) > 1 and parts[1] else 10
        self.match = re.compile(parts[2] if len(parts) > 2 else ".")
        self.path = os.path.join(out, "perf.data")
        self.proc = None
        self.names = {}

    def poll(self, now, pid):
        if self.proc or now < self.start:
            return
        threads = {tid: t[0] for tid, t in thread_times(pid).items()
                   if self.match.search(t[0])}
        if not threads:
            return
        self.names = threads
        self.pid = pid
        cmd = ["perf", "record", "-q", "-m", "16", "-e", "cpu-clock:u", "-F", "4000",
               "-t", ",".join(map(str, threads)), "-o", self.path,
               "--proc-map-timeout", "60000", "--", "sleep", str(self.secs)]
        if os.environ.get("DRUN_PERF_CALLGRAPH"):
            cmd[cmd.index("4000")] = "1000"
            cmd[2:2] = ["--call-graph", "dwarf,16384"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def function_starts(self, modmap):
        """Per module, the sorted function starts from its .eh_frame_hdr
        binary-search table, read out of the live process."""
        import bisect
        import struct
        self.starts = {}
        try:
            mem = open(f"/proc/{self.pid}/mem", "rb")
        except OSError:
            return
        for base, size, name, hdr in modmap:
            if not hdr:
                continue
            try:
                mem.seek(hdr)
                head = mem.read(12)
                ver, ptr_enc, cnt_enc, tbl_enc = head[:4]
                # The common layout: pcrel sdata4 pointer, udata4 count,
                # datarel sdata4 table.
                if ver != 1 or cnt_enc != 0x03 or tbl_enc != 0x3b:
                    continue
                count = struct.unpack_from("<I", head, 8)[0]
                if not 0 < count < 4_000_000:
                    continue
                table = mem.read(count * 8)
            except (OSError, ValueError):
                continue
            locs = sorted(hdr + struct.unpack_from("<i", table, i * 8)[0]
                          for i in range(len(table) // 8))
            self.starts[name] = locs
        mem.close()
        self.bisect = bisect

    def locate(self, ip, modmap):
        for base, size, name, _ in modmap:
            if base <= ip < base + size:
                locs = getattr(self, "starts", {}).get(name)
                if locs:
                    k = self.bisect.bisect_right(locs, ip) - 1
                    if k >= 0:
                        fn = locs[k]
                        return (f"{name}+{fn - base:#x}  (fn, +{ip - fn:#x} hot)",
                                name, f"{name}+{ip - base:#x}")
                return f"{name}+{ip - base:#x}", name, None
        return None, None, None

    def report(self, modmap, top=25):
        if not self.proc:
            print("\nperf: the run ended before sampling started")
            return
        self.proc.wait()
        out = subprocess.run(["perf", "script", "-i", self.path, "-F",
                              "tid,ip,sym,dso"], capture_output=True,
                             text=True).stdout
        spots = collections.Counter()
        hot_ip = collections.defaultdict(collections.Counter)
        where = collections.Counter()
        per_thread = collections.defaultdict(collections.Counter)
        total = 0
        for line in out.splitlines():
            m = re.match(r"\s*(\d+)\s+([0-9a-f]+)\s+(.*)\s+\(([^()]*)\)\s*$",
                         line)
            if not m:
                continue
            tid, ip, sym, dso = int(m.group(1)), int(m.group(2), 16), \
                m.group(3).strip(), m.group(4)
            spot, module, exact = self.locate(ip, modmap)
            if exact:
                # Group by function; keep the hottest instruction of each.
                spot = spot.split("  (fn,")[0]
                hot_ip[spot][exact] += 1
            if spot:
                area = "guest " + module
            elif sym and sym != "[unknown]":
                spot = f"{sym}  ({os.path.basename(dso)})"
                area = "host " + os.path.basename(dso)
            elif dso.startswith("/tmp/perf-"):
                # Anonymous executable memory: llvmpipe's shader JIT when the
                # present runs headless, or our own Xbyak stubs.
                spot = f"{ip:#x}  (jit)"
                area = "host jit (llvmpipe, stubs)"
            else:
                spot = f"{ip:#x}  ({os.path.basename(dso)})"
                area = "unknown " + os.path.basename(dso)
            spots[spot] += 1
            where[area] += 1
            per_thread[self.names.get(tid, str(tid))][area] += 1
            total += 1
        if not total:
            print("\nperf: no samples (perf_event_paranoid, or no such thread)")
            return
        print(f"\n== perf {self.path}  {total} samples over "
              f"{len(self.names)} threads ({self.secs:.0f} s)")
        print("  by area:")
        for area, n in where.most_common(10):
            print(f"    {n / total * 100:5.1f}%  {area}")
        print("  hot spots (guest functions by .eh_frame start; disassemble "
              "with tools/scedis.py):")
        for spot, n in spots.most_common(top):
            hot = ""
            if spot in hot_ip:
                ip, c = hot_ip[spot].most_common(1)[0]
                hot = f"   hottest {ip} {c / n * 100:.0f}%"
            print(f"    {n / total * 100:5.1f}%  {spot[:90]}{hot}")
        print("  by thread:")
        for name, areas in sorted(per_thread.items(),
                                  key=lambda kv: -sum(kv[1].values()))[:10]:
            n = sum(areas.values())
            main = ", ".join(f"{a} {c / n * 100:.0f}%"
                             for a, c in areas.most_common(3))
            print(f"    {name[:16]:16} {n / total * 100:5.1f}%  {main}")


def report_capture(cap_dir):
    tool = os.path.join(ROOT, "tools", "gpu_capture.py")
    out = subprocess.run([sys.executable, tool, "summary", cap_dir],
                         capture_output=True, text=True).stdout
    if not out:
        print(f"\ncapture: none in {cap_dir}")
        return
    lines = out.splitlines()
    cut = next((i for i, l in enumerate(lines) if "SAMPLED" in l), len(lines))
    print("\n== capture " + cap_dir)
    for l in lines[:min(cut, 30)]:
        print("  " + l)
    m = re.search(r"scanout\s+(0x[0-9a-f]+)", out)
    if m:
        pngs = [f for f in os.listdir(cap_dir) if f"_rt_{m.group(1)}_" in f]
        print("  scanout png: " + (os.path.join(cap_dir, pngs[0]) if pngs
                                   else "(not dumped)"))
    print(f"  shaders: {cap_dir}/frame_*_sh_<addr>.s   "
          f"more: tools/gpu_capture.py draws|draw N|graph {cap_dir}")


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
    p.add_argument("--capture", default="",
                   help="GPU frame capture into OUT/cap, then exit and "
                        "summarize: t=SECS, busy=DRAWS or f=FRAME")
    p.add_argument("--tracy", default="",
                   help="START[:SECS]: record a Tracy trace from START s into "
                        "OUT/trace.tracy and print where the time went")
    p.add_argument("--tracy-top", type=int, default=30,
                   help="zones listed in the Tracy report")
    p.add_argument("--perf", default="",
                   help="START:SECS[:THREAD_REGEX]: perf-sample the matching "
                        "threads (all by default) and print hot guest "
                        "module+offsets and host functions")
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
    cap_dir = os.path.join(a.out, "cap")
    if a.capture:
        kind, _, val = a.capture.partition("=")
        knob = {"t": "DELTA_GPU_CAPTURE_AFTER", "busy": "DELTA_GPU_CAPTURE_BUSY",
                "f": "DELTA_GPU_CAPTURE"}.get(kind)
        if not knob or not val:
            p.error("--capture takes t=SECS, busy=DRAWS or f=FRAME")
        shutil.rmtree(cap_dir, ignore_errors=True)
        os.makedirs(cap_dir)
        opts += [f"{knob}={val}", f"DELTA_GPU_CAPTURE_DIR={cap_dir}",
                 "DELTA_GPU_CAPTURE_EXIT=1"]
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
    tracy = Tracy(a.tracy, a.out) if a.tracy else None
    perf = Perf(a.perf, a.out) if a.perf else None
    stall_probes = 0
    buf = b""
    try:
        while True:
            now = time.time() - run.started
            # A capture being written is not cut off by -t (a killed
            # capture is truncated JSON); it exits the process when done.
            capturing = a.capture and run.capture_started and \
                not run.capture_done and now < a.time + 120
            if now > a.time and not capturing:
                break
            if tracy:
                tracy.poll(now, proc.pid)
            if perf:
                perf.poll(now, proc.pid)
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
        if tracy:
            tracy.threads_end = thread_times(proc.pid)
        if perf and perf.proc:
            perf.function_starts(run.modmap)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        subprocess.run(["pkill", "-9", "-x", "ps4delta"])
        log.close()
    run.report(proc.poll(), log_path)
    if tracy:
        tracy.report(a.tracy_top)
    if perf:
        perf.report(run.modmap)
    if a.capture:
        report_capture(cap_dir)
    if a.film:
        out = film(film_dir, os.path.join(a.out, "film.png"))
        print(f"\nfilm: {out}" if out else "\nfilm: (no frames)")


if __name__ == "__main__":
    main()
