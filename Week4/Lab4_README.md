# Lab 4 – Logical Reasoning for Planning

**Idea:** Logic + Search = Planning.
**Files:** `Lab4_Planning.ipynb` (all code, tests, answers), `planner.pl` (Prolog extension), this README.
**Run:** open the notebook, run all cells top to bottom (Python 3, standard library only). Prolog is optional; if SWI-Prolog (`swipl`) is not installed, the notebook falls back to a Python emulation of the same facts and rules.

---

## Steps followed

### Step 1 – Specify the problem (Task 0)
- **I** = {At(Robot,A), At(Package,A)}; **G** = {At(Package,C)}.
- Actions: `Move(x,y)` for A-B, B-A, B-C, C-B; `PickUp(Package,l)`; `Drop(Package,l)`.

| Action | Preconditions | Adds | Deletes |
|---|---|---|---|
| Move(x,y) | At(Robot,x) | At(Robot,y) | At(Robot,x) |
| PickUp(Package,l) | At(Robot,l), At(Package,l) | Holding(Package) | At(Package,l) |
| Drop(Package,l) | At(Robot,l), Holding(Package) | At(Package,l) | Holding(Package) |

- `PickUp(Package,A)` **is applicable** in I: both At(Robot,A) and At(Package,A) hold.
- `Drop(Package,C)` **is not**: the robot is not at C and is not holding the package.
- Being in the action list does not make an action applicable; every precondition must hold in the current state (S |= Preconditions(a)).

### Step 2 – Solve by hand (Task 1)
| State | Facts | Action taken |
|---|---|---|
| S0 | At(Robot,A), At(Package,A) | – |
| S1 | At(Robot,A), Holding(Package) | PickUp(Package,A) |
| S2 | At(Robot,B), Holding(Package) | Move(A,B) |
| S3 | At(Robot,C), Holding(Package) | Move(B,C) |
| S4 | At(Robot,C), At(Package,C) | Drop(Package,C) |

**Catch in the lab sheet:** its hint (`Move(A,B)`, `PickUp(Package,B)`, ...) is invalid, because the package is at A, not B. Doing the plan by hand first is what exposes this. The notebook also proves it (Test E).

### Step 3 – Build the planner (Task 2)
Following the specification in the lab prompt:
- A **state** is a `frozenset` of proposition strings.
- An **Action** holds name, positive/negative preconditions, positive/negative effects.
- `applicable(state, a)`: all positive preconditions are in the state and no negative precondition is.
- `apply_action(state, a)`: remove negative effects, then add positive effects.
- `bfs_plan(...)`: breadth-first search over states with a queue and a `parent` map, which doubles as the visited set. It stops when the goal set is a subset of the state, and returns "no plan" when the queue is empty.

| Idea | Where in code |
|---|---|
| Preconditions | `applicable()` |
| Effects | `apply_action()` |
| Goal | `goal <= nxt` in `bfs_plan` |
| BFS | `deque` frontier in `bfs_plan` |

### Step 4 – Test it (Task 3)
| Test | Setup | Result | Valid? |
|---|---|---|---|
| A – solvable | original problem | PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | Yes, replayed by an independent validator |
| B – impossible | PickUp removed | **No plan found** | Correct |
| C1 – irrelevant actions | goal At(Robot,C) | Move(A,B), Move(B,C); package still at A | Correct: robot at C ≠ package at C |
| C2 – irrelevant actions | goal At(Package,C), Move actions only | **No plan found** | Correct |
| D – goal already true | goal At(Package,A) | empty plan | Correct |
| E – lab-hint plan | Move(A,B), PickUp(Package,B), ... | rejected at step 2, missing At(Package,B) | Correctly invalid |
| F – optimality | original problem | length 4 | BFS returns a shortest plan |

The validator replays the plan and checks every precondition without using the search, so it acts as a second, independent check.

### Step 5 – Explain how logic and search combine (Task 4)
```
Current state
  -> Check action preconditions (S |= Pre(a))     LOGIC
  -> Keep only applicable actions                 (the "?" in the lab)
  -> Generate successor state S' = Apply(S, a)    LOGIC
  -> Search over alternatives (BFS queue)         SEARCH
  -> Goal reached?  yes: return plan / no: repeat
```
**Logic determines what is possible; search determines what to try.** Logic alone says what is allowed but cannot pick a sequence; search alone would try impossible actions.

### Step 6 – Check the plan independently (Task 5)
The notebook prints, for each step, its preconditions and whether they hold in the state where it runs (all true for the BFS plan).
**Trust the independently executed state transitions over the LLM's explanation.** They are computed mechanically. An LLM explanation is generated text and can sound right while skipping a precondition. A generated explanation is not independent verification.

### Step 7 – Optional Prolog (Tasks 6–8)
`planner.pl` contains the `connected/2` facts, `can_move/2`, `valid_move/2`, a `valid_plan/1` checker, and the wet-road rules.

| Query | Result | Why |
|---|---|---|
| `can_move(a,b)` | true | fact `connected(a,b)` + rule |
| `can_move(a,c)` | false | no fact or rule gives `connected(a,c)` |
| `valid_move(a,b)`, `valid_move(b,c)` | true | facts exist |
| `valid_move(a,c)` | false | rejects the proposed `Move(a,c)` |
| `valid_plan([a,b,c])` | true | each step is valid |
| `valid_plan([a,c])` | false | step a→c unsupported |
| `reduce_speed` | true | chain below |

- **Task 6(c):** `can_move(X,Y) :- connected(X,Y)` is the implication Connected(X,Y) → CanMove(X,Y) written backwards (`:-` means "if").
- **Task 8:** wet_road (fact) ⇒ slippery :- wet_road (rule) ⇒ reduce_speed :- slippery (rule) ⇒ reduce_speed (conclusion).

> Note: SWI-Prolog was not available in the environment where I tested, so the Prolog queries were verified through the Python emulation in the notebook. Run `swipl planner.pl` and try the queries yourself to confirm.

---

## Reflection answers
1. **Why specify preconditions/effects first?** The LLM can only code what you define precisely. The spec also gives you the ground truth to test against. Otherwise you cannot tell whether the code is right.
2. **Error without precondition checks:** the robot could pick up the package from B when it is at A, or drop it while not holding it, producing a "plan" that is physically impossible.
3. **Why "looks reasonable" is not valid:** the lab-hint plan reads naturally but fails at step 2. Validity needs every precondition to hold, not plausibility.
4. **LLM contribution:** boilerplate (data structures, BFS loop, printing) and explanations. (Edit this to describe what *your* LLM actually produced and what you changed.)
5. **Verified independently:** the plan by hand, the applicability rules, impossible/irrelevant-action tests, and a separate validator replaying each step.
6. **Where logic is used:** checking preconditions against the state, applying add/delete effects, and testing whether the goal holds. In the Prolog part, deriving `can_move`/`valid_move` from facts and rules.
7. **Link to search:** planning is state-space search. States are sets of facts, actions are transitions (only where preconditions hold), cost is 1 per action, and the goal is a set of facts. BFS here is the same blind search studied before.
8. **Prolog facts vs rules:** a fact is unconditionally true (`connected(a,b).`); a rule is conditional (`head :- body`).
9. **Query vs entailment:** a query asks whether something follows from the knowledge base.
10. **Why an independent verifier:** a separate logical checker does not share the generator's mistakes, so it catches invalid LLM output instead of repeating it.
