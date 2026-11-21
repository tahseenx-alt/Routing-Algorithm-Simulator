# Routing Algorithm Simulator and Performance Comparator

A desktop application for a Computer Networks mini project. You draw a network
of routers and links, run five routing algorithms on it, watch them work step
by step, send a packet from a source to a destination, and get a fair
comparison that explains which algorithm is best for that network and why.

Built with Python, Tkinter and Matplotlib only. It runs fully offline on
Windows, macOS and Linux.

## Features

- **Network editor:** add, move and delete routers (A–O, up to 15), add and
  delete links, set link delays (1–100 ms), fail and restore links. Five sample
  networks are included, plus a random connected network generator (seeded).
  Networks are saved and loaded as JSON, and an invalid file gives a friendly
  error.
- **Five algorithms:** Dijkstra (link state, like OSPF), Bellman-Ford (distance
  vector by delay, a RIP-style process), Hop Count (the RIP metric), Flooding
  and Random routing.
- **Step-by-step visualisation:** the current router is yellow, finalised
  routers are green, and the path is thick blue. Live distance labels sit next
  to each router. Step, Back, Play, Pause and Reset buttons, a speed slider,
  and a text log of every step.
- **Packet simulation:** a dot moves along the links with a running ms timer,
  and the Timeline tab lists every event. You can schedule
  "fail link X–Y at t ms", with rerouting ON or OFF.
- **Run All & Compare:** a comparison table, three charts (delivery time,
  total messages for K packets, computation time), a "best algorithm" banner
  with a plain-English explanation, and CSV/PNG export for the report.
- **Failure Test:** fail a link and see how every algorithm recovers (new path,
  new time, extra routing messages, rounds to reconverge).
- **Count-to-Infinity Demo:** the linear A–B–C network with and without split
  horizon, and the triangle loop where split horizon is not enough.

## Installation and running

```bash
cd routing_simulator
pip install -r requirements.txt
python main.py        # start the application
pytest                # run the test suite (about 500 tests, a few seconds)
```

Tkinter is part of the python.org installers for Windows and macOS. On
Ubuntu/Debian, install it with `sudo apt install python3-tk python3-pil.imagetk`.
The second package is needed by Matplotlib's Tk backend.

## Project structure

```
routing_simulator/
├── main.py                 entry point
├── core/                   network model, settings, packet simulation, metrics, JSON storage
├── algorithms/             the five algorithms + shared distance-vector engine (no Tkinter)
├── gui/                    Tkinter windows, canvas, panels, Matplotlib charts
├── samples/                small, medium, large, count_to_infinity, triangle_loop
└── tests/                  pytest tests (NetworkX is used only here, as a reference)
```

Algorithm logic never imports Tkinter. Each algorithm returns a
`RoutingResult` containing a list of `Step`s, and the GUI only replays those
steps.

## The algorithms and the CN concepts behind them

| Algorithm | CN concept | How it works here |
|---|---|---|
| **Dijkstra** | Link state routing (OSPF) | Every router floods one Link State Advertisement (LSA), so all routers know the full map. Each router then runs Dijkstra's Shortest Path First (SPF) algorithm with a priority queue (`heapq`). Routing overhead = the number of LSA transmissions, which we simulate. |
| **Bellman-Ford** | Distance vector routing (RIP-style process) | Routers know only their neighbours. Every round, each router sends its distance vector to each neighbour and recomputes `D_x(y) = min over v [cost(x,v) + D_v(y)]`. This repeats until nothing changes. Routing overhead = the number of vectors sent. Split horizon is optional. |
| **Hop Count** | The RIP metric (infinity = 16) | Chooses the path with the fewest hops and ignores delay. Ties are broken by lower delay. Delivery time still uses the real delays, so it can be slower than Dijkstra. Routing overhead = the vectors sent by the distance-vector engine using the hop metric. |
| **Flooding** | Flooding with duplicate suppression | Each router forwards the first copy it receives on every link except the incoming one, and drops duplicates. It is simulated with a time-ordered event queue, so the first copy really is the fastest one. It needs no tables, but sends many copies for every packet. |
| **Random** | Random walk forwarding (no routing tables) | Forwards to a random neighbour, and avoids going straight back unless the router is a dead end. TTL = 2 × routers. Runs 20 seeded trials and reports success rate, average delivery time and average transmissions per trial. |

## Assumptions

1. **Timing model.** The source sends at t = 0 with no processing. Every
   router that receives the packet, including the destination, spends `p` ms
   processing it (p is set in the GUI, default 1 ms). Therefore
   **delivery time = sum of link delays + p × number of hops**. To make the
   shortest-path algorithms find the truly fastest path, a link's
   **effective cost is `delay_ms + p`**. If you change p, rerun the algorithms.
2. **Infinity.**
   - Hop metric: infinity = 16, exactly as in RIP. The 15-router limit keeps
     every path under 16 hops.
   - Delay metric: infinity = (sum of the effective costs of **all** links,
     active or failed) + 1. This is larger than any real path, so a long route
     (for example a single 18 ms link) is never mistaken for "unreachable",
     and count-to-infinity still ends.
   - Distance vector stops after `max_rounds` (500) and reports "did not
     converge".
3. **Synchronous distance-vector rounds.** In every round all routers send
   their vectors at the same time (one vector to one neighbour = one message),
   then all routers recompute from scratch. `rounds` counts up to the last
   round in which any table changed. Messages are counted for every round that
   was run, including the final round with no change.
4. **Two kinds of overhead.**
   - *Routing overhead* (one-time setup): LSA transmissions for Dijkstra,
     vectors sent for Bellman-Ford and Hop Count, and 0 for Flooding and
     Random.
   - *Per-packet transmissions*: the number of hops for path-based algorithms,
     all copies for Flooding, and the average per trial for Random.
   - **Total messages for K packets = routing overhead + K × per-packet
     transmissions**. K is set in the GUI (default 100).
5. **Best-algorithm rule.** Only algorithms that fully delivered the packet are
   eligible; Random counts only with a 100 % success rate. The lowest delivery
   time wins. On a tie, the fewest total messages for K packets wins. On a
   further tie, the lowest computation time wins (the average of 100
   `time.perf_counter` runs, where one Random run = all 20 trials).
   Computation times vary slightly between runs, so a tie broken by time can
   change from one run to the next.
6. **Link failure during a transfer.** A copy that is on the link when it fails
   is lost. A packet that has not reached the link yet stops at the router
   before it. With rerouting OFF, the packet is "Not delivered". With rerouting
   ON, it waits at the last router it reached for the **reconvergence delay**,
   then continues on the recomputed path:
   - Distance vector (Bellman-Ford, Hop Count): rounds to reconverge ×
     `update_period_ms` (at least one period).
   - Link state (Dijkstra): 1 × `update_period_ms`, since one LSA flood plus an
     SPF recomputation is treated as one update period.
   - Flooding and Random: no reconvergence. Other copies or random choices just
     continue.
7. **Warm-start reconvergence.** After a failure, the two routers at the ends
   of the link notice at once and recompute from the vectors they still have.
   The other routers continue from their old tables. This is what makes
   count-to-infinity visible; restarting from scratch would hide it.
8. **Failure Test overhead.** For Dijkstra, the two routers at the ends of the
   failed link each flood a new LSA, and both floods are counted. For distance
   vector, every vector sent during reconvergence is counted. LSA flooding is
   counted within the source's connected part of the network. Distance vector
   runs on every router.

## Limitations

- Split horizon only stops loops between two neighbours. Loops through three or
  more routers (see the triangle demo) can still count to infinity.
- Poison reverse, triggered updates, hold-down timers and route timeouts are
  not implemented.
- Distance-vector rounds are synchronous; real RIP routers send updates at
  unsynchronised times (every 30 s).
- Links are symmetric (the same delay in both directions). There is no
  queueing, bandwidth or packet loss apart from scheduled link failures.

## Viva demo script (about 10 minutes)

1. **Load a sample.** Choose `small` in the Network box and press
   **Load Sample**. Point out routers, link delays and the settings
   (p = 1 ms, K = 100).
2. **Visualize Dijkstra.** Set Algorithm = Dijkstra, Source = A,
   Destination = E, then press **Visualize Selected** and **Play**. Explain:
   yellow = current router, green = finalised, purple labels = distances, blue
   = the chosen path. Note that costs are delay + p. Click router B to show
   its routing table.
3. **Run All & Compare.** Explain the table and the three charts. Dijkstra,
   Bellman-Ford and Flooding all reach E in the same time, because they all
   find the fastest path. **Flooding loses on total messages**: it needs no
   setup, but sends 9 copies for every packet, so 100 packets cost 900
   messages, against about 250 for Dijkstra. Random routing is not reliable
   (success < 100 %), so it is not eligible. Export the CSV/PNG for the report.
4. **Fail a link.** Tick "Fail link", choose C-E, set t = 6 ms and press
   **Send Packet**. The copy on C–E is lost, the packet waits for
   reconvergence and is rerouted (switch Rerouting off to show
   "Not delivered"). Then press **Failure Test** and fail C–E: compare the
   extra routing messages (LSA floods versus distance vectors) and the rounds
   to reconverge.
5. **Count-to-Infinity Demo.**
   - Tab 1, linear A–B–C with B–C failed: without split horizon, B's cost to C
     counts 3, 5, 7 … up to 16 (14 rounds). With split horizon, it converges in
     1 round.
   - Tab 2, the triangle: even with split horizon, the costs to D still count
     to 16, because split horizon only stops loops between two neighbours.

## Tests

`pytest` checks, among other things:

- Dijkstra and Bellman-Ford distances equal NetworkX
  (`single_source_dijkstra_path_length` / `single_source_bellman_ford_path_length`,
  edge weight = delay + p) on all samples and 20 random networks, for p = 0 and 2.
- The infinity bug regression: on `large.json` (links > 16 ms), Bellman-Ford
  matches NetworkX and no route is marked unreachable.
- Hop count equals `single_source_shortest_path_length`, and flooding delivery
  time equals Dijkstra's for p = 0, 1 and 3.
- Random routing is reproducible with the same seed and differs with another
  seed.
- Count-to-infinity on the linear and triangle samples, with and without split
  horizon.
- Packet timing (sum of delays + p × hops), and failure during transfer with
  rerouting OFF and ON.
- The best-selection rules, the edge cases (same source and destination,
  disconnected network, destination unreachable after a failure), and save →
  load round trips plus invalid files.
