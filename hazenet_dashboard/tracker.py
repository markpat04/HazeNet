"""
HazeNet training tracker — ให้ dashboard ติดตามการเทรน real-time.

วิธีใช้ (เพิ่ม ~4 บรรทัดใน train_operator_m2.py / sweep_m2.py):

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hazenet_dashboard"))
    from tracker import Run

    run = Run(model="CLNOLowRank", config={"rank":32,"hidden":64,"lr":1e-3})
    for ep in range(EPOCHS):
        ...
        run.log_epoch(ep, train=tr_loss, test=te_loss, lr=cur_lr)
    run.finish(metrics={"MAE": mae_te, "RMSE": rmse_te}, status="done")

เขียนลง hazenet_dashboard/runs/<timestamp>/  (meta.json + progress.jsonl)
ไม่ต้องมี dependency ใดๆ.
"""
import os, json, time

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


class Run:
    def __init__(self, model="model", config=None, note=""):
        self.id   = time.strftime("%Y%m%d-%H%M%S")
        self.dir  = os.path.join(_DIR, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.t0   = time.time()
        self.prog = os.path.join(self.dir, "progress.jsonl")
        self.meta = dict(id=self.id, model=model, config=config or {},
                         note=note, status="running",
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._save_meta()

    def _save_meta(self):
        json.dump(self.meta, open(os.path.join(self.dir, "meta.json"), "w",
                                  encoding="utf-8"), ensure_ascii=False, indent=2)

    def log_epoch(self, epoch, train=None, test=None, lr=None, **extra):
        rec = dict(epoch=int(epoch), t=round(time.time() - self.t0, 1))
        if train is not None: rec["train"] = float(train)
        if test  is not None: rec["test"]  = float(test)
        if lr    is not None: rec["lr"]    = float(lr)
        rec.update({k: float(v) for k, v in extra.items()})
        with open(self.prog, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.meta["last_epoch"] = int(epoch)
        self._save_meta()

    def finish(self, metrics=None, status="done"):
        self.meta["status"]  = status
        self.meta["metrics"] = {k: round(float(v), 3) for k, v in (metrics or {}).items()}
        self.meta["elapsed_s"] = round(time.time() - self.t0, 1)
        self.meta["finished"]  = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_meta()
