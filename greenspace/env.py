"""GreenSpaceEnv: the grid environment both sides of the project build on.

It exposes two interfaces on purpose:

  1. The standard Gymnasium interface (reset / step / action_space /
     observation_space).

  2. A set of plain helper methods (clone / place / remove / current_objective /
     valid_placements) that local search agents use directly, because local
     search wants to evaluate whole candidate boards, not crawl one step at a time.

Both views share the same underlying state, so an agent written against either
interface keeps working when the other side's code is merged in.

If gymnasium is installed the env is a real gymnasium.Env. If not, it falls back
to a tiny shim with the same surface so the code still runs.
"""

from __future__ import annotations

import numpy as np

from .grid import (
    Board,
    Cell,
    GREEN_TYPES,
    LEGAL_BASE,
    generate_board,
    render_ascii,
)
from .cost import CostConfig, objective, total_spent
from .pollution import simulate


# --- gymnasium with graceful fallback -------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # gymnasium not installed -> minimal shim, same surface
    _HAS_GYM = False

    class _Env:
        pass

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)

    class _SpacesShim:
        Box = _Box
        Discrete = _Discrete

    class _GymShim:
        Env = _Env

    gym = _GymShim()
    spaces = _SpacesShim()
# --------------------------------------------------------------------------


# action layout: a flat Discrete index encodes (cell, green_type)
# action = cell_index * len(GREEN_TYPES) + green_slot
def _decode_action(action: int, n_cols: int, total_cells: int):
    k = len(GREEN_TYPES)
    cell = action // k
    slot = action % k
    r, c = cell // n_cols, cell % n_cols
    return r, c, GREEN_TYPES[slot]


class GreenSpaceEnv(gym.Env):
    """Optimize green infrastructure placement to cut pollution under a budget."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        board: Board | None = None,
        budget: float = 20.0,
        cost_config: CostConfig | None = None,
        # used only when board is None (env generates its own)
        m: int = 12,
        n: int = 12,
        n_infra: int | None = None,
        n_drains: int = 2,
        seed: int | None = None,
    ):
        super().__init__()
        self._init_kwargs = dict(m=m, n=n, n_infra=n_infra, n_drains=n_drains)
        self._provided_board = board
        self.budget = float(budget)
        self.cfg = cost_config or CostConfig()
        self._seed = seed

        # set up a board now so spaces are defined
        self.board = board.copy() if board is not None else generate_board(
            seed=seed, **self._init_kwargs
        )
        self.m, self.n = self.board.shape
        self.total_cells = self.m * self.n

        self.placements = np.full((self.m, self.n), -1, dtype=int)
        self.spent = 0.0

        # observation: stacked feature channels
        #   0: base==EMPTY, 1: base==INFRA, 2: base==DRAIN
        #   3: emission (raw), 4: has_placement, 5: remaining-budget broadcast
        #   6+: any 2D feature map found in board.extra (added automatically)
        self._base_channels = 6
        self.n_channels = self._base_channels + len(self._extra_channel_keys())
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(self.n_channels, self.m, self.n), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.total_cells * len(GREEN_TYPES))

    # ---- gymnasium interface ---------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if _HAS_GYM:
            super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        if self._provided_board is not None:
            self.board = self._provided_board.copy()
        else:
            self.board = generate_board(seed=self._seed, **self._init_kwargs)
        self.m, self.n = self.board.shape
        self.placements = np.full((self.m, self.n), -1, dtype=int)
        self.spent = 0.0
        return self.observation(), self._info()

    def step(self, action: int):
        r, c, green_type = _decode_action(int(action), self.n, self.total_cells)
        before = self.current_objective()
        ok = self.place(r, c, green_type)

        if not ok:
            # invalid action: no-op with a small penalty, keep episode going
            obs, reward, term, trunc, info = self.observation(), -0.1, False, self._done(), self._info()
            info["valid"] = False
            return obs, reward, term, trunc, info

        after = self.current_objective()
        reward = before - after   # objective dropped -> positive reward
        terminated = self._done()
        info = self._info()
        info["valid"] = True
        return self.observation(), float(reward), terminated, False, info

    def render(self):
        return render_ascii(self.board.base, self.placements)

    # ---- local-search helper interface -----------------------------------
    def clone(self) -> "GreenSpaceEnv":
        c = GreenSpaceEnv(
            board=self.board, budget=self.budget, cost_config=self.cfg,
            seed=self._seed, **self._init_kwargs,
        )
        c.board = self.board.copy()
        c.placements = self.placements.copy()
        c.spent = self.spent
        return c

    def remaining_budget(self) -> float:
        return self.budget - self.spent

    def can_place(self, r: int, c: int, green_type: Cell) -> bool:
        if self.placements[r, c] != -1:
            return False
        if self.board.base[r, c] != LEGAL_BASE[green_type]:
            return False
        return self.cfg.placement_cost(green_type) <= self.remaining_budget() + 1e-9

    def place(self, r: int, c: int, green_type: Cell) -> bool:
        if not self.can_place(r, c, green_type):
            return False
        self.placements[r, c] = int(green_type)
        self.spent += self.cfg.placement_cost(green_type)
        return True

    def remove(self, r: int, c: int) -> bool:
        p = self.placements[r, c]
        if p < 0:
            return False
        self.spent -= self.cfg.placement_cost(Cell(p))
        self.placements[r, c] = -1
        return True

    def valid_placements(self) -> list[tuple[int, int, Cell]]:
        out = []
        for green_type in GREEN_TYPES:
            target = LEGAL_BASE[green_type]
            cost = self.cfg.placement_cost(green_type)
            if cost > self.remaining_budget() + 1e-9:
                continue
            rs, cs = np.where((self.board.base == target) & (self.placements == -1))
            for r, c in zip(rs, cs):
                out.append((int(r), int(c), green_type))
        return out

    def current_placements(self) -> list[tuple[int, int, Cell]]:
        rs, cs = np.where(self.placements >= 0)
        return [(int(r), int(c), Cell(self.placements[r, c])) for r, c in zip(rs, cs)]

    def pollution(self) -> float:
        return simulate(
            self.board.base, self.placements, self.board.emission, self.board.flow,
            retention=self.cfg.retention, edge_runoff_counts=self.cfg.edge_runoff_counts,
            runoff_coeff=self.board.extra.get("runoff_coeff"),
            storm_depth=float(self.board.extra.get("storm_depth", 1.0)),
        )

    def current_objective(self) -> float:
        pollution = self.pollution()
        context = {
            "pollution": pollution,
            "spent": self.spent,
            "placements": self.placements,
            "board": self.board,
            "env": self,
        }
        return objective(pollution, self.spent, self.cfg, context)

    def baseline_pollution(self) -> float:
        """Pollution with zero green infrastructure (the do-nothing case)."""
        empty = np.full((self.m, self.n), -1, dtype=int)
        return simulate(
            self.board.base, empty, self.board.emission, self.board.flow,
            retention=self.cfg.retention, edge_runoff_counts=self.cfg.edge_runoff_counts,
            runoff_coeff=self.board.extra.get("runoff_coeff"),
            storm_depth=float(self.board.extra.get("storm_depth", 1.0)),
        )

    # ---- internals --------------------------------------------------------
    def _done(self) -> bool:
        return len(self.valid_placements()) == 0

    def _extra_channel_keys(self) -> list[str]:
        """Keys in board.extra that are 2D maps matching the grid, in stable order.

        Anything Shaurya attaches to board.extra as an (m, n) array shows up as an
        extra observation channel with no other code change.
        """
        keys = []
        for k in sorted(self.board.extra):
            v = self.board.extra[k]
            if isinstance(v, np.ndarray) and v.shape == self.board.base.shape:
                keys.append(k)
        return keys

    def observation(self) -> np.ndarray:
        base = self.board.base
        channels = [
            (base == Cell.EMPTY).astype(np.float32),
            (base == Cell.INFRA).astype(np.float32),
            (base == Cell.DRAIN).astype(np.float32),
            self.board.emission.astype(np.float32),
            (self.placements >= 0).astype(np.float32),
            np.full(base.shape, self.remaining_budget(), dtype=np.float32),
        ]
        for k in self._extra_channel_keys():
            channels.append(self.board.extra[k].astype(np.float32))
        return np.stack(channels).astype(np.float32)

    # kept for backward compatibility
    def _obs(self) -> np.ndarray:
        return self.observation()

    def _info(self) -> dict:
        return {
            "pollution": self.pollution(),
            "spent": self.spent,
            "remaining_budget": self.remaining_budget(),
            "objective": self.current_objective(),
            "baseline_pollution": self.baseline_pollution(),
        }
