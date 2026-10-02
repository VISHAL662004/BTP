"""Progress bars.

In an interactive terminal a live `tqdm` bar is shown. When output is redirected (log files,
background jobs) tqdm's carriage-return updates make logs unreadable, so a plain-text bar is
written as a normal line every `step_pct` percent instead, e.g.

    epochs [██████░░░░░░░░░░░░░░░░░░]  5/20  25% | elapsed 3m25s ETA 10m16s | val_cpm=0.568
"""
import sys
import time


def text_bar(done: int, total: int, width: int = 24) -> str:
    frac = 0.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    full = int(round(frac * width))
    return "█" * full + "░" * (width - full)


def _fmt(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Progress:
    def __init__(self, iterable=None, total=None, desc: str = "", leave: bool = True, unit: str = "it",
                 step_pct: int = 10, quiet_when_redirected: bool = False):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)
        self.desc, self.step_pct, self.done = desc, step_pct, 0
        self.quiet = quiet_when_redirected and not sys.stderr.isatty()
        self.t0, self.postfix, self._next, self._closed = time.time(), {}, step_pct, False
        self.bar = None
        if sys.stderr.isatty():
            from tqdm import tqdm
            self.bar = tqdm(total=self.total, desc=desc, leave=leave, unit=unit, dynamic_ncols=True)

    def __iter__(self):
        try:
            for x in self.iterable:
                yield x
                self.update(1)
        finally:
            self.close()

    def set_postfix(self, **kw):
        self.postfix.update(kw)
        if self.bar is not None:
            self.bar.set_postfix(**kw, refresh=False)

    def update(self, n: int = 1):
        self.done += n
        if self.bar is not None:
            self.bar.update(n)
        elif not self.quiet:
            pct = 100 * self.done / max(self.total, 1)
            if pct >= self._next or self.done >= self.total:
                self._emit()
                while self._next <= pct:
                    self._next += self.step_pct

    def _emit(self):
        el = time.time() - self.t0
        eta = el * (self.total - self.done) / max(self.done, 1)
        post = " | " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.postfix.items()) if self.postfix else ""
        w = len(str(self.total))
        print(f"{self.desc} [{text_bar(self.done, self.total)}] {self.done:>{w}}/{self.total} "
              f"{100 * self.done / max(self.total, 1):3.0f}% | elapsed {_fmt(el)} ETA {_fmt(eta)}{post}",
              file=sys.stderr, flush=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.bar is not None:
            self.bar.close()
