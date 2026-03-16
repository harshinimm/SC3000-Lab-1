import json
import math
import heapq
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================
# Part 1: NYC shortest path with energy budget
# ============================================================
# Assignment facts used here:
# - start node = '1'
# - goal node = '50'
# - energy budget for tasks 2 and 3 = 287932
# - G is adjacency list, Coord gives node coordinates,
#   Dist['u,v'] gives distance, Cost['u,v'] gives energy cost
# ============================================================

START_NODE = "1"
GOAL_NODE = "50"
ENERGY_BUDGET = 287932

# ============================================================
# Part 2: Grid world
# ============================================================
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)
BLOCKS = {(2, 1), (2, 3)} 
ACTIONS = ["U", "D", "L", "R"]
GAMMA = 0.9
STEP_REWARD = -1
GOAL_REWARD = 10
MC_EPSILON = 0.1
Q_EPSILON = 0.1
Q_ALPHA = 0.1


# ============================================================
# General helpers
# ============================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nyc_instance(
    g_path: str = "G.json",
    coord_path: str = "Coord.json",
    dist_path: str = "Dist.json",
    cost_path: str = "Cost.json",
):
    G_raw = load_json(g_path)
    Coord_raw = load_json(coord_path)
    Dist_raw = load_json(dist_path)
    Cost_raw = load_json(cost_path)

    G = {str(k): [str(x) for x in v] for k, v in G_raw.items()}
    Coord = {str(k): (float(v[0]), float(v[1])) for k, v in Coord_raw.items()}
    Dist = {str(k): float(v) for k, v in Dist_raw.items()}
    Cost = {str(k): float(v) for k, v in Cost_raw.items()}
    return G, Coord, Dist, Cost


def edge_key(u: str, v: str) -> str:
    return f"{u},{v}"


def reconstruct_path(parent: Dict, end):
    if end not in parent:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def path_distance_and_energy(path: List[str], Dist: Dict[str, float], Cost: Dict[str, float]):
    total_dist = 0.0
    total_cost = 0.0
    for i in range(len(path) - 1):
        k = edge_key(path[i], path[i + 1])
        total_dist += Dist[k]
        total_cost += Cost[k]
    return total_dist, total_cost


def print_part1_result(title: str, path: Optional[List[str]], total_dist: float, total_cost: float, time_taken: float):
    print(f"\n[{title}]")
    if not path:
        print("Shortest path: No feasible path found.")
        return
    print("Shortest path:", "->".join(path))
    print(f"Shortest distance: {total_dist:.6f}")
    print(f"Total energy cost: {total_cost:.6f}")
    print(f"Time taken: {time_taken:.6f} seconds")


# ============================================================
# Part 1 Task 1: UCS without energy constraint
# ============================================================
def ucs_task1(G, Dist, Cost, start=START_NODE, goal=GOAL_NODE):
    time_start = time.perf_counter()
    pq = [(0.0, start)]
    best_dist = {start: 0.0}
    parent = {start: None}

    while pq:
        d, u = heapq.heappop(pq)
        if d > best_dist.get(u, float("inf")):
            continue
        if u == goal:
            break

        for v in G.get(u, []):
            new_dist = d + Dist[edge_key(u, v)]
            if new_dist < best_dist.get(v, float("inf")):
                best_dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    path = reconstruct_path(parent, goal)
    if not path:
        return None, float("inf"), float("inf")
    total_dist, total_cost = path_distance_and_energy(path, Dist, Cost)
    time_end = time.perf_counter()
    time_taken = time_end - time_start
    return path, total_dist, total_cost,time_taken


# ============================================================
# Part 1 Task 2: UCS with energy constraint
# State = (node, used_energy)
# We keep non-dominated labels for pruning.
# ============================================================
def is_dominated(labels, new_dist, new_energy):
    for d, e in labels:
        if d <= new_dist and e <= new_energy:
            return True
    return False

# Keeps an old label unless the new label is better or equal in both distance and energy.
def add_label(labels, new_dist, new_energy):
    kept = []
    for dist, energy in labels:
        if not (new_dist <= dist and new_energy <= energy):
            kept.append((dist, energy))
    kept.append((new_dist, new_energy))
    return kept


def ucs_task2(G, Dist, Cost, budget=ENERGY_BUDGET, start=START_NODE, goal=GOAL_NODE):
    time_start = time.perf_counter()
    pq = [(0.0, 0.0, start)]  # (distance, energy, node)
    parent = {(start, 0.0): None}
    labels = defaultdict(list)
    labels[start].append((0.0, 0.0))

    best_goal_state = None
    best_goal_dist = float("inf")

    while pq:
        dist_so_far, energy_so_far, node = heapq.heappop(pq)

        # skip if we already have a better path to the goal
        if dist_so_far > best_goal_dist:
            continue
        
        # check if we reached the goal 
        if node == goal:
            if dist_so_far < best_goal_dist:
                best_goal_dist = dist_so_far
                best_goal_state = (node, energy_so_far)
            continue
        
        # expand neighbors
        for v in G.get(node, []):
            k = edge_key(node, v)
            # compute new distance and energy
            new_dist = dist_so_far + Dist[k]
            new_energy = energy_so_far + Cost[k]

            # skip if energy exceeds budget
            if new_energy > budget:
                continue

            # skip if this path is dominated by an existing label for v
            if is_dominated(labels[v], new_dist, new_energy):
                continue
            
            # add new label and push to queue
            labels[v] = add_label(labels[v], new_dist, new_energy)
            parent[(v, new_energy)] = (node, energy_so_far)
            heapq.heappush(pq, (new_dist, new_energy, v))

    if best_goal_state is None:
        return None, float("inf"), float("inf")

    # reconstruct 
    rev = []
    cur = best_goal_state
    while cur is not None:
        rev.append(cur[0])
        cur = parent[cur]
    path = rev[::-1]
    total_dist, total_cost = path_distance_and_energy(path, Dist, Cost)
    time_end = time.perf_counter()
    time_taken = time_end - time_start
    return path, total_dist, total_cost, time_taken


# ============================================================
# Part 1 Task 3: A* with energy constraint
# Heuristic: Euclidean distance between coordinates.
# This is admissible for road distance if straight-line distance
# never overestimates actual shortest travel distance.
# ============================================================
# hypotenuse is shortest distance between two points with third side 
def euclidean_heuristic(node: str, goal: str, Coord: Dict[str, Tuple[float, float]]) -> float:
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.hypot(x1 - x2, y1 - y2)


def astar_task3(G, Coord, Dist, Cost, budget=ENERGY_BUDGET, start=START_NODE, goal=GOAL_NODE):
    time_start = time.perf_counter()
    start_h = euclidean_heuristic(start, goal, Coord)
    pq = [(start_h, 0.0, 0.0, start)]  # (f, g (distance), energy, node)
    parent = {(start, 0.0): None}
    labels = defaultdict(list)
    labels[start].append((0.0, 0.0))

    best_goal_state = None
    best_goal_dist = float("inf")

    while pq:
        f, g, energy_so_far, u = heapq.heappop(pq)

        # skip if we already have a better path to the goal
        if g > best_goal_dist:
            continue

        # check if we reached the goal
        if u == goal:
            if g < best_goal_dist:
                best_goal_dist = g
                best_goal_state = (u, energy_so_far)
            continue

        for v in G.get(u, []):
            k = edge_key(u, v)
            ng = g + Dist[k]
            ne = energy_so_far + Cost[k]
            if ne > budget:
                continue

            if is_dominated(labels[v], ng, ne):
                continue

            labels[v] = add_label(labels[v], ng, ne)
            parent[(v, ne)] = (u, energy_so_far)
            nf = ng + euclidean_heuristic(v, goal, Coord)
            heapq.heappush(pq, (nf, ng, ne, v))

    if best_goal_state is None:
        return None, float("inf"), float("inf")

    rev = []
    cur = best_goal_state
    while cur is not None:
        rev.append(cur[0])
        cur = parent[cur]
    path = rev[::-1]
    total_dist, total_cost = path_distance_and_energy(path, Dist, Cost)
    time_end = time.perf_counter()
    time_taken = time_end - time_start
    return path, total_dist, total_cost, time_taken


# ============================================================
# Part 2: Grid world environment
# ============================================================
# We represent the grid world as a class to create environment
class GridWorld:
    def __init__(self):
        self.width = GRID_SIZE
        self.height = GRID_SIZE
        self.start = START_STATE
        self.goal = GOAL_STATE
        self.blocks = set(BLOCKS)
        self.actions = list(ACTIONS)

    # to check if a position is blocked
    def states(self):
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.blocks
        ]

    # check if a state is terminal (i.e. goal state)
    def is_terminal(self, s):
        return s == self.goal

    def move(self, s, a):
        if self.is_terminal(s):
            return s

        #valid movements
        x, y = s
        if a == "U":
            ns = (x, y + 1)
        elif a == "D":
            ns = (x, y - 1)
        elif a == "L":
            ns = (x - 1, y)
        elif a == "R":
            ns = (x + 1, y)
        else:
            raise ValueError(f"Unknown action: {a}")

        # check bounds and blocks
        if (
            ns[0] < 0 or ns[0] >= self.width or
            ns[1] < 0 or ns[1] >= self.height or
            ns in self.blocks
        ):
            return s
        return ns

    # reward function: +10 for reaching goal, -1 otherwise
    def reward(self, s, a, ns):
        return GOAL_REWARD if ns == self.goal else STEP_REWARD

    def perpendicular_actions(self, a):
        if a in ("U", "D"):
            return ["L", "R"]
        return ["U", "D"]

    def transitions(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]

        # primary action with 0.8 probability, perpendicular actions with 0.1 each
        perp_act1, perp_act2 = self.perpendicular_actions(action)
        candidates = [
            (0.8, self.move(state, action)),
            (0.1, self.move(state, perp_act1)),
            (0.1, self.move(state, perp_act2)),
        ]

        agg = defaultdict(float)
        for prob, next_state in candidates:
            agg[next_state] += prob

        out = []
        for next_state, prob in agg.items():
            out.append((prob, next_state, self.reward(state, action, next_state)))
        return out

    # Returns next state, reward, and whether episode ended.
    def sample_step(self, state, action):
        trans = self.transitions(state, action)
        r = random.random()
        cumilative_prob = 0.0
        for p, ns, rew in trans:
            cumilative_prob += p
            if r <= cumilative_prob:
                done = self.is_terminal(ns)
                return ns, rew, done
        ns, rew = trans[-1][1], trans[-1][2]
        return ns, rew, self.is_terminal(ns)


# ============================================================
# Policy / value display helpers
# ============================================================
def best_action_from_q(qvals: Dict[str, float]):
    return max(ACTIONS, key=lambda a: qvals.get(a, 0.0))

# formats the policy dictionary into a grid string for display
def format_policy(policy: Dict[Tuple[int, int], str]) -> str:
    rows = []
    for y in range(GRID_SIZE - 1, -1, -1):
        row = []
        for x in range(GRID_SIZE):
            s = (x, y) #
            if s in BLOCKS:
                row.append(" X ")
            elif s == GOAL_STATE:
                row.append(" G ")
            else:
                row.append(f" {policy.get(s, '?')} ")
        rows.append("".join(row))
    return "\n".join(rows)


def print_policy(policy, title):
    print(f"\n{title}")
    print(format_policy(policy))


def print_values(V, title):
    print(f"\n{title}")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = []
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in BLOCKS:
                row.append("  X   ")
            else:
                row.append(f"{V.get(s, 0.0):6.2f}")
        print(" ".join(row))


# ============================================================
# Part 2 Task 1: Value iteration and policy iteration
# ============================================================
def value_iteration(env: GridWorld, gamma=GAMMA, theta=1e-8):
    states = env.states()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        new_V = V.copy()
        for s in states:
            if env.is_terminal(s):
                new_V[s] = 0.0
                continue
            action_values = []
            for a in ACTIONS:
                q = 0.0
                for p, ns, r in env.transitions(s, a):
                    q += p * (r + gamma * V[ns])
                action_values.append(q)
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break

    policy = {}
    for s in states:
        if env.is_terminal(s):
            continue
        best_a = None
        best_q = -float("inf")
        for a in ACTIONS:
            q = 0.0
            for p, ns, r in env.transitions(s, a):
                q += p * (r + gamma * V[ns])
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return V, policy


def policy_evaluation(env: GridWorld, policy, gamma=GAMMA, theta=1e-8):
    states = env.states()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        new_V = V.copy()
        for s in states:
            if env.is_terminal(s):
                new_V[s] = 0.0
                continue
            a = policy[s]
            val = 0.0
            for p, ns, r in env.transitions(s, a):
                val += p * (r + gamma * V[ns])
            new_V[s] = val
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
    return V


def policy_iteration(env: GridWorld, gamma=GAMMA, theta=1e-8):
    states = env.states()
    policy = {s: "U" for s in states if not env.is_terminal(s)}

    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        stable = True
        for s in states:
            if env.is_terminal(s):
                continue
            old_a = policy[s]
            best_a = old_a
            best_q = -float("inf")
            for a in ACTIONS:
                q = 0.0
                for p, ns, r in env.transitions(s, a):
                    q += p * (r + gamma * V[ns])
                if q > best_q:
                    best_q = q
                    best_a = a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        if stable:
            return V, policy


# ============================================================
# Part 2 Task 2: Monte Carlo control
# ============================================================
def epsilon_greedy_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[state][a])


def mc_control(env: GridWorld, episodes=5000, gamma=GAMMA, epsilon=MC_EPSILON):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for _ in range(episodes):
        episode = []
        s = env.start
        while True:
            a = epsilon_greedy_action(Q, s, epsilon)
            ns, r, done = env.sample_step(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break

        G = 0.0
        seen = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in seen:
                seen.add((s, a))
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

    policy = {}
    for s in env.states():
        if not env.is_terminal(s):
            policy[s] = max(ACTIONS, key=lambda a: Q[s][a])
    return Q, policy


# ============================================================
# Part 2 Task 3: Q-learning
# ============================================================
def q_learning(env: GridWorld, episodes=5000, gamma=GAMMA, epsilon=Q_EPSILON, alpha=Q_ALPHA):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    for _ in range(episodes):
        s = env.start
        while True:
            a = epsilon_greedy_action(Q, s, epsilon)
            ns, r, done = env.sample_step(s, a)
            best_next = 0.0 if done else max(Q[ns].values())
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])
            s = ns
            if done:
                break

    policy = {}
    for s in env.states():
        if not env.is_terminal(s):
            policy[s] = max(ACTIONS, key=lambda a: Q[s][a])
    return Q, policy


# ============================================================
# Main
# ============================================================
def run_part1():
    G, Coord, Dist, Cost = load_nyc_instance()

    path1, d1, c1, t1 = ucs_task1(G, Dist, Cost)
    print_part1_result("Part 1 - Task 1 (UCS without energy constraint)", path1, d1, c1, t1)

    path2, d2, c2, t2 = ucs_task2(G, Dist, Cost)
    print_part1_result("Part 1 - Task 2 (UCS with energy budget)", path2, d2, c2, t2)

    path3, d3, c3, t3 = astar_task3(G, Coord, Dist, Cost)
    print_part1_result("Part 1 - Task 3 (A* with energy budget)", path3, d3, c3, t3)



def run_part2():
    env = GridWorld()

    V_vi, pi_vi = value_iteration(env)
    print_values(V_vi, "Part 2 - Task 1: Value Iteration Values")
    print_policy(pi_vi, "Part 2 - Task 1: Value Iteration Policy")

    V_pi, pi_pi = policy_iteration(env)
    print_values(V_pi, "Part 2 - Task 1: Policy Iteration Values")
    print_policy(pi_pi, "Part 2 - Task 1: Policy Iteration Policy")

    Q_mc, pi_mc = mc_control(env, episodes=5000)
    print_policy(pi_mc, "Part 2 - Task 2: Monte Carlo Policy")

    Q_ql, pi_ql = q_learning(env, episodes=5000)
    print_policy(pi_ql, "Part 2 - Task 3: Q-Learning Policy")


if __name__ == "__main__":
    run_part1()
    run_part2()
