# Lab 3 – Search and A* (Warehouse Robot)

**Files:** `Lab3_AStar.ipynb` (code + tests), `README.md` (this file), `LLM_Prompt.txt` (reusable prompt).
**Run:** open the notebook in Jupyter / VS Code and run all cells (Python 3, no extra libraries).

---

## Task 0 – Problem formulation
| Component | Specification |
|---|---|
| State | Robot position `(row, col)` on a free cell |
| Actions | Up, Down, Left, Right |
| Transition | Move one cell in the chosen direction if it is inside the map and not `#` |
| Initial state | `S` = (1, 1) |
| Goal | `G` = (7, 15) |
| Cost | 1 per move |

- **(a)** Only the robot's position. The map is fixed, so it is not part of the state.
- **(b)** Moving into `#`, or outside the map.
- **(c)** Yes. The same action in the same state always gives the same result.
- **(d)** A sequence of valid actions from `S` to `G`. The optimal one has cost 40.

## Task 1 – Design
State = tuple `(r, c)`. Map = list of lists. Valid actions = 4 directions filtered by bounds and `#`. Goal test = `state == goal`. Frontier = priority queue (`heapq`) of `(f, h, counter, state)`, plus `g` and `parent` dicts. Path = follow `parent` back from goal and reverse. Report: found, path, length, expanded.

## Task 3 – Test results
| Test | Expected | Result |
|---|---|---|
| Warehouse | path found | Found, length **40**, **63** expanded |
| Trivial (`#SG##`) | length 1 | Pass |
| No solution | report failure, no loop | Pass (9 expanded, `found=False`) |
| Alternative paths | length equals BFS (shortest) | Pass (length 6) |

## Task 4 – Where concepts appear
| Concept | In code |
|---|---|
| State | `(row, col)` tuples |
| Action | `ACTIONS` dict |
| Transition | `valid_moves()` |
| Goal test | `if s == goal` |
| g(n) | `g` dict, `new_g = g[s] + 1` |
| h(n) | `manhattan()` called as `h(nxt, goal)` |
| f(n) | `new_g + hn`, first item in the heap tuple |
| Frontier | `frontier` heap |
| Visited | `closed` set |
| Path reconstruction | `parent` dict + `reconstruct()` |

- **(a)** A binary min-heap (`heapq`).
- **(b)** `heappop` returns the entry with the lowest f; ties go to the lower h.
- **(c)** When a neighbour is pushed: `hn = h(nxt, goal)`.
- **(d)** Yes: `new_g + hn`, stored as the heap's priority.
- **(e)** The `closed` set skips states already expanded, and a neighbour is only re-queued if it has a strictly lower g.

## Task 5 – BFS vs A*
| Measure | BFS | A* |
|---|---|---|
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

- **(a)** Yes. **(b)** Yes, both length 40 (both optimal).
- **(c)** Neither: both expanded 63 of the 64 free cells. The warehouse is a winding corridor, and the goal sits right beside the start (in straight-line terms) behind a wall. Manhattan distance points into walls, so it cannot help.
- **(d)** In general A* expands fewer states because h lets it favour states that look closer to the goal. On an open room it expands 16 states vs 72 for BFS. Here the maze structure removes that advantage.

## Task 6 – Heuristic investigation
Results on three maps (length / expanded):

| Heuristic | Warehouse (opt 40) | Open room (opt 16) | Trap map (opt 12) |
|---|---|---|---|
| h = 0 | 40 / 63 | 16 / 72 | 12 / 32 |
| Manhattan | 40 / 63 | 16 / 16 | 12 / 17 |
| Euclidean | 40 / 63 | 16 / 57 | 12 / 14 |
| 2 × Manhattan | 40 / 63 | 16 / 16 | **14** / 14 |

- **h = 0:** still finds the optimal path, but A* degenerates into blind (uniform-cost) search and expands the most states.
- **Euclidean:** still optimal (it never overestimates), but it is a weaker estimate than Manhattan, so it expands more states on open maps.
- **2 × Manhattan:** fewest expansions but can return a **non-optimal** path (14 instead of 12 on the trap map). It overestimates, so it breaks admissibility h(n) ≤ h*(n).
- **Takeaway:** an admissible heuristic keeps A* optimal; a more informed (closer to h*) one expands fewer states; an over-optimistic-in-cost (overestimating) one trades optimality for speed.

## Task 7 – Evaluating the LLM (adapt to your own experience)
1. State, `valid_moves`, parent-based path reconstruction and the `heapq` frontier were correct straight away.
2. Design issue: with plain `(f, counter)` ordering, A* expanded all 72 cells on the open room, no better than BFS. Ties on f were broken in FIFO order.
3. Found by comparing expansion counts with BFS on the open room.
4. Possibly `heapq`, tie-breaker counter, and "closed set".
5. Yes: added `h` as a tie-breaker (`(f, h, counter, state)`).
6. The no-solution test (catches infinite loops) and the BFS-length comparison (catches non-optimal paths).
7. No. Output looked plausible even when expansion behaviour was poor; only comparison tests exposed it.
8. A* is only as good as its heuristic, and on a maze it can be as slow as BFS.

## Final reflection
1. **Formulate first:** the formulation defines state, actions and goal. Without it you cannot tell whether code is right, and the algorithm has nothing to be correct against.
2. **Informed:** A* uses problem knowledge (h) to choose what to expand next, not just the order of discovery.
3. **Heuristic matters:** it decides both optimality (admissible or not) and efficiency (how close to h*). Experiments above show 16 vs 72 expansions and a 14 vs 12 path.
4. **LLM contribution:** fast boilerplate, explanations and test ideas; the engineer supplied the design, tests and judgement.
5. **Risk of not testing:** silent bugs such as non-optimal paths, infinite loops on unsolvable maps, or hidden inefficiency, all of which can look like "working output".
