"""
Toy testbed for "Plan Slowly, Act Fast".

A 2D point robot must reach a goal in 50 steps. A hidden context (mass, friction,
motor orientation, wind) changes by blocks of episodes and comes back later.

Building blocks:
  - world model: Bayesian linear regression, one per context or a single one with forgetting
  - context inference: prediction error over a short window
  - planner: MPC by CEM inside the world model (expensive)
  - habit: linear policy distilled from the planner's decisions (cheap)
  - arbiter: plans when surprise is high or when the habit is not yet competent

Agents compared:
  A. planif_seul       permanent planner, single world model with forgetting
  B. habitude_figee    learns the habit during the first block, then acts by habit only
  C. arbitre_sans_mem  arbiter, single world model and habit with forgetting
  D. arbitre_mem_ctx   arbiter, world model and habit stored per context

Computation is counted in "model steps" (one step = one prediction s -> s').

Usage:  python sim.py [n_seeds]     (default 5, numpy only, writes results.json)
"""
import json
import sys
import time

import numpy as np

# ---------------------------------------------------------------- environment
DT, T, NOISE, GAIN = 0.1, 50, 0.005, 4.0
CONTEXTS = {
    0: dict(name="C0 nominal", m=1.0, f=0.10, theta=0.0, wind=(0.0, 0.0)),
    1: dict(name="C1 heavy load, motors rotated by 90 deg", m=2.0, f=0.10,
            theta=np.pi / 2, wind=(0.0, 0.0)),
    2: dict(name="C2 wind, commands rotated by -120 deg", m=1.0, f=0.20,
            theta=-2 * np.pi / 3, wind=(0.4, 0.0)),
}
SCHEDULE = [0, 1, 0, 2, 1, 0, 2]
EP_PER_BLOCK = 20
SUCCESS_RADIUS = 0.15


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def true_dynamics(c):
    p = CONTEXTS[c]
    i2 = np.eye(2)
    r = rot(p["theta"])
    w = np.array(p["wind"])
    a = np.zeros((4, 4)); b = np.zeros((4, 2)); k = np.zeros(4)
    a[2:, 2:] = (1 - p["f"]) * i2
    b[2:] = DT * GAIN / p["m"] * r
    k[2:] = DT * w
    a[:2, :2] = i2
    a[:2, 2:] = DT * (1 - p["f"]) * i2
    b[:2] = DT * DT * GAIN / p["m"] * r
    k[:2] = DT * DT * w
    return a, b, k


TRUE = {c: true_dynamics(c) for c in CONTEXTS}


def env_step(s, act, c, rng):
    a, b, k = TRUE[c]
    return a @ s + b @ act + k + rng.normal(0, NOISE, 4)


# ---------------------------------------------------------------- building blocks
class BayesLinear:
    """Bayesian linear regression y = phi @ W, with an optional forgetting factor."""

    def __init__(self, d_in, d_out, lam=1e-3, gamma=1.0):
        self.P = lam * np.eye(d_in)
        self.Q = np.zeros((d_in, d_out))
        self.gamma, self.n, self._W = gamma, 0.0, None

    def update(self, phi, y):
        if self.gamma < 1.0:
            self.P *= self.gamma
            self.Q *= self.gamma
            self.n *= self.gamma
        self.P += np.outer(phi, phi)
        self.Q += np.outer(phi, y)
        self.n += 1.0
        self._W = None

    @property
    def W(self):
        if self._W is None:
            self._W = np.linalg.solve(self.P, self.Q)
        return self._W

    def copy_weak(self, scale):
        m = BayesLinear(self.P.shape[0], self.Q.shape[1], gamma=self.gamma)
        m.P = self.P * scale + 1e-3 * np.eye(self.P.shape[0])
        m.Q = self.Q * scale
        m.n = 0.0
        return m


def wm_features(s, a):
    return np.concatenate([s, a, [1.0]])


def wm_error(model, s, a, s2):
    """Prediction error normalised by the noise (about 1 when the model is right)."""
    pred = wm_features(s, a) @ model.W
    return float(np.sum((s2 - pred) ** 2) / (4 * NOISE ** 2))


def hab_features(s, g):
    return np.concatenate([g - s[:2], s[2:], [1.0]])


H, K, ITERS, N_ELITE = 15, 128, 3, 16
PLAN_COST = H * K * ITERS  # model steps per planned decision


def cem_plan(W, s, g, rng, mean=None):
    if mean is None:
        mean = np.zeros((H, 2))
    std = np.full((H, 2), 0.6)
    a_s, a_a, a_b = W[:4], W[4:6], W[6]
    for _ in range(ITERS):
        U = np.clip(mean + std * rng.standard_normal((K, H, 2)), -1, 1)
        S = np.repeat(s[None], K, 0)
        cost = np.zeros(K)
        for t in range(H):
            S = S @ a_s + U[:, t] @ a_a + a_b
            cost += np.sum((S[:, :2] - g) ** 2, 1) + 0.02 * np.sum(S[:, 2:] ** 2, 1) \
                + 0.01 * np.sum(U[:, t] ** 2, 1)
        elite = U[np.argsort(cost)[:N_ELITE]]
        mean, std = elite.mean(0), elite.std(0) + 0.05
    return mean[0], mean


def step_cost(S, g, u):
    return float(np.sum((S[:2] - g) ** 2) + 0.02 * np.sum(S[2:] ** 2) + 0.01 * np.sum(u ** 2))


def seq_cost(W, s, g, U):
    """Cost predicted by the world model for an action sequence (open loop)."""
    c = 0.0
    for u in U:
        s = s @ W[:4] + u @ W[4:6] + W[6]
        c += step_cost(s, g, u)
    return c


def habit_cost(W, Wh, s, g):
    """Cost predicted by the world model if the habit drives (closed loop)."""
    c = 0.0
    for _ in range(H):
        u = np.clip(hab_features(s, g) @ Wh, -1, 1)
        s = s @ W[:4] + u @ W[4:6] + W[6]
        c += step_cost(s, g, u)
    return c


class Context:
    def __init__(self, model, gamma_h=1.0):
        self.model = model
        self.habit = BayesLinear(5, 2, lam=1e-2, gamma=gamma_h)
        # predicted cost gap between habit and plan (value of computation); 1 = not competent yet
        self.disagree = 1.0


# ---------------------------------------------------------------- agent
# Thresholds are hand-tuned (see the limits in experiments/toy_testbed/README.md).
TAU_SURPRISE, TAU_DISAGREE, N_MIN, P_CHECK = 4.0, 0.15, 60, 0.05
TAU_CTX, TAU_NEW, WINDOW, YOUNG = 6.0, 15.0, 5, 100


class Agent:
    def __init__(self, kind, pretrained, rng):
        self.kind, self.rng = kind, rng
        mem = kind == "arbitre_mem_ctx"
        # habit: same forgetting for both arbiters; with context memory it only applies to
        # the active context, so inactive contexts are preserved.
        # world model: single with forgetting (adapts) or one per context without (remembers)
        g_wm = 1.0 if mem else 0.98
        g_h = 0.99
        m = pretrained.copy_weak(1.0)
        m.gamma = g_wm
        self.g_h = g_h
        self.contexts = [Context(m, g_h)]
        self.cur = 0
        self.surprise = 1.0
        self.buf = []
        self.unexplained = 0
        self.since_new = 10 ** 6
        self.frozen = False
        self.plan_mean = None

    # --- decision
    def act(self, s, g):
        ctx = self.contexts[self.cur]
        evals = 1  # world-model prediction used for the surprise
        if self.kind == "planif_seul":
            plan = True
        elif self.frozen:
            plan = False
        else:
            trusted = ctx.habit.n >= N_MIN and ctx.disagree < TAU_DISAGREE
            plan = (self.surprise > TAU_SURPRISE) or (not trusted) or \
                (self.rng.random() < P_CHECK)
        a_hab = None
        if ctx.habit.n > 0:
            a_hab = np.clip(hab_features(s, g) @ ctx.habit.W, -1, 1)
        if plan:
            a, mean = cem_plan(ctx.model.W, s, g, self.rng, self.plan_mean)
            self.plan_mean = np.vstack([mean[1:], mean[-1:]])
            evals += PLAN_COST
            if not self.frozen and self.kind != "planif_seul":
                f = hab_features(s, g)
                if a_hab is not None:
                    # value of computation: how much better the plan is than the habit,
                    # according to the world model
                    j_plan = seq_cost(ctx.model.W, s, g, mean)
                    j_hab = habit_cost(ctx.model.W, ctx.habit.W, s, g)
                    gap = max(0.0, j_hab - j_plan) / (j_plan + 0.1)
                    ctx.disagree = 0.9 * ctx.disagree + 0.1 * min(gap, 2.0)
                    evals += 2 * H
                ctx.habit.update(f, a)
            return a, evals, True
        self.plan_mean = None
        if a_hab is None:
            a_hab = np.zeros(2)
        return a_hab, evals, False

    # --- world-model learning and context inference
    def observe(self, s, a, s2):
        ctx = self.contexts[self.cur]
        err = wm_error(ctx.model, s, a, s2)
        self.surprise = 0.7 * self.surprise + 0.3 * err
        evals = 0
        if self.kind == "arbitre_mem_ctx":
            self.buf = (self.buf + [(s, a, s2)])[-WINDOW:]
            self.since_new += 1
            errs = []
            for c in self.contexts:
                if c.model.n >= 20 or c is ctx:
                    errs.append(np.mean([wm_error(c.model, *tr) for tr in self.buf]))
                else:
                    errs.append(np.inf)
                evals += len(self.buf)
            best = int(np.argmin(errs))
            if errs[best] < TAU_CTX:
                self.cur, self.unexplained = best, 0
            elif errs[best] > TAU_NEW:
                self.unexplained += 1
                young = self.contexts[self.cur].model.n < YOUNG
                # the whole window must be unexplained before a context is created
                if self.unexplained > WINDOW and self.since_new > 40 and not young:
                    src = self.contexts[self.cur].model
                    # very weak prior (about 3 samples): new data dominate quickly
                    new = Context(src.copy_weak(3.0 / max(src.P[-1, -1], 1.0)), self.g_h)
                    self.contexts.append(new)
                    self.cur, self.unexplained, self.since_new = len(self.contexts) - 1, 0, 0
                    self.surprise = TAU_SURPRISE + 1
            ctx = self.contexts[self.cur]
            # modulated plasticity: an established model only learns what it already explains
            if ctx.model.n < 300 or wm_error(ctx.model, s, a, s2) < 25:
                ctx.model.update(wm_features(s, a), s2)
        else:
            ctx.model.update(wm_features(s, a), s2)
        return evals


# ---------------------------------------------------------------- experiment
def pretrain(rng, n=300):
    m = BayesLinear(7, 4)
    for _ in range(n):
        s = np.concatenate([rng.uniform(-1, 1, 2), rng.uniform(-0.5, 0.5, 2)])
        a = rng.uniform(-1, 1, 2)
        m.update(wm_features(s, a), env_step(s, a, 0, rng))
    return m


def episodes(seed):
    rng = np.random.default_rng(1000 + seed)
    out = []
    for c in SCHEDULE:
        for _ in range(EP_PER_BLOCK):
            while True:
                s0, g = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
                if np.linalg.norm(s0 - g) > 0.5:
                    break
            out.append((c, s0, g))
    return out


def run(kind, seed):
    rng = np.random.default_rng(seed)
    agent = Agent(kind, pretrain(np.random.default_rng(500 + seed)), rng)
    log = dict(success=[], compute=[], plan_frac=[], ctx=[], n_ctx=[])
    for i, (c, s0, g) in enumerate(episodes(seed)):
        if kind == "habitude_figee" and i == EP_PER_BLOCK:
            agent.frozen = True
        s = np.concatenate([s0, [0.0, 0.0]])
        comp, nplan = 0, 0
        agent.plan_mean = None
        for _ in range(T):
            a, ev, planned = agent.act(s, g)
            s2 = env_step(s, a, c, rng)
            comp += ev + (0 if agent.frozen else agent.observe(s, a, s2))
            nplan += planned
            s = s2
        log["success"].append(bool(np.linalg.norm(s[:2] - g) < SUCCESS_RADIUS))
        log["compute"].append(comp)
        log["plan_frac"].append(nplan / T)
        log["ctx"].append(c)
        log["n_ctx"].append(len(agent.contexts))
    return log


AGENTS = ["planif_seul", "habitude_figee", "arbitre_sans_mem", "arbitre_mem_ctx"]

if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results = {}
    for kind in AGENTS:
        results[kind] = []
        for sd in range(seeds):
            t0 = time.time()
            results[kind].append(run(kind, sd))
            r = results[kind][-1]
            print(f"{kind:18s} seed {sd}: success {np.mean(r['success']):.2f} "
                  f"mean compute {np.mean(r['compute']):,.0f} "
                  f"contexts {r['n_ctx'][-1]}  ({time.time() - t0:.1f}s)", flush=True)
    meta = dict(schedule=SCHEDULE, ep_per_block=EP_PER_BLOCK,
                contexts={k: v["name"] for k, v in CONTEXTS.items()}, plan_cost=PLAN_COST)
    with open("results.json", "w") as fh:
        json.dump(dict(meta=meta, results=results), fh)
