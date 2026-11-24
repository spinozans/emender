"""S16: keep only the checked descriptor across four groups. Fresh m20 on every wave."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter_ns as clock

from .branch_domain_bench import Meter, WavePages, WaveTap, main, worker
from .branch_image import ImagePages, P100Image
from .serving_verify import HERE, require
from .terminal_branch_bench import WireMeter

GATES = 148
ORDER = ("fresh-A", "image-A", "image-B", "fresh-B")
PRIOR = HERE / "receipts/memory-program-s15/fresh-A"


class ImageWavePages(WaveTap, WireMeter, ImagePages):
    pass


class DescriptorMeter(Meter):
    def __init__(self, man):
        super().__init__(man)
        self.image = None
        self.image_summary = None
        self.starts = []

    @contextmanager
    def schedule(self, label):
        require(label in ORDER, "unknown descriptor schedule")
        groups = [(i,) for i in range(4)]
        if label.startswith("fresh"):
            yield groups
            return
        t = clock()
        with P100Image(self.man, scope="whole", max_waves=4) as im:
            self.image = im
            opened = clock() - t
            yield groups
            finish = clock()
        closed = clock() - finish
        require(im.closed and im.started == im.reaped == im.waves == 4, "unclosed descriptor panel")
        self.image = None
        self.image_summary = dict(
            open_ns=opened,
            close_ns=closed,
            full_verifications=im.attestations,
            checks=im.checks,
            starts_bytes=im.starts_bytes,
            offsets_checked=im.offsets_checked,
            workers_started=im.started,
            workers_reaped=im.reaped,
            closed=im.closed,
        )

    def factory(self, **binding):
        require(self.pages is None and len(self.opens) == len(self.starts), "prior m20 not reaped")
        before = clock()
        p = (ImageWavePages if self.image is not None else WavePages)(**binding)
        try:
            self.starts.append(
                dict(
                    start_ns=before,
                    pid=p.proc.pid,
                    attest_ns=p.attest_ns,
                    workspace_ns=p.workspace_ns,
                    root=p.root,
                    nodes=p.nodes,
                    documents=len(p.desc.starts),
                    starts_bytes=p.desc.starts.nbytes,
                )
            )
        except BaseException:
            p.close()
            raise
        self.pages = p
        return p

    def domain(self, count):
        require(count == 1, "one native process per wave")
        if self.image is None:
            return super().domain(count)
        return self.image.wave("P100", self.man["mount"], scope="whole", pages_type=self.factory)

    @contextmanager
    def open(self, count):
        with super().open(count) as pair:
            yield pair
        self.starts[-1]["reaped_ns"] = clock()

    def extra_cell(self):
        return dict(image=self.image_summary, workspace_starts=self.starts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    args = ap.parse_args()
    if args.cell:
        worker(args.root, args.cell, gates=GATES, meter_type=DescriptorMeter)
    else:
        main(args.root, gates=GATES, order=ORDER, module=__spec__.name, compare_to=PRIOR)
