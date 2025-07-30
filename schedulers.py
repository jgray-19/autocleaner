import math

from torch.optim.lr_scheduler import _LRScheduler


class HalvingCosineLR(_LRScheduler):
    """
    • Cosine decay:  hi ─► B   in N steps
    • Cosine bounce: B  ─► hi*0.75 in N steps
      (total mini-cycle = 2 N; peak is reduced to 0.75 every 2 N steps)

    Pass  `last_epoch = initial_step-1`  when resuming from a checkpoint
    so that the very next .step() continues the curve seamlessly.
    """

    def __init__(self, optimizer, a: float, b: float, n: int, last_epoch: int = -1):
        self.A = float(a)
        self.B = float(b)
        self.N = int(n)
        self.cycle_len = 2 * n  # N ↓ + N ↑
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _cosine(start, end, t, T):
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t / T))

    def _lr_at_step(self, step):
        cycle = step // self.cycle_len  # 0-based mini-cycle index
        hi = self.A * (0.75**cycle)  # peak of this cycle (reduced to 0.75 each cycle)
        phase_step = step % self.cycle_len  # position within the cycle

        if phase_step < self.N:  # decay hi → B
            return self._cosine(hi, self.B, phase_step, self.N)

        # bounce B → hi*0.75
        t = phase_step - self.N
        hi_next = hi * 0.75
        return self._cosine(self.B, hi_next, t, self.N)

    def get_lr(self):
        step = self.last_epoch + 1  # LR to apply *after* this .step()
        lr = self._lr_at_step(step)
        return [lr for _ in self.base_lrs]
