# 2022/2023, symulacje komputerowe.
# Projekt nr 4 - "Pożar Lasu", Konrad Pomichowski, 150900.

import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation, colors, patches

# Source: http://nifty.stanford.edu/2007/shiflet-fire/
# Source: https://www.bib.irb.hr/278897/download/278897.Ljiljana_Bodrozic_ceepus2006_2.pdf
# Source: https://en.wikipedia.org/wiki/Von_Neumann_neighborhood

TREE_NEIGHBORS = (
    (-1, 0),  # East
    (0, -1),  # North
    (1, 0),  # West
    (0, 1),  # South
)
DIRECTIONS = ["S", "E", "N", "W"]

# Initialization of the possible states in the cellular automaton.
NUM_STATES = 5
POSSIBLE_STATES = tuple(range(NUM_STATES))

# Empty place -> 0, healthy tree -> 1, burning tree (fire) -> 2, burnt tree -> 3, water -> 4
EMPTY_PLACE, HEALTHY_TREE, BURNING_TREE, BURNT_TREE, WATER = POSSIBLE_STATES

# Iterations of the simulation.
GRID_SIZE = 35
M, N = (GRID_SIZE, GRID_SIZE)  # Forest grid size.


# Initialize the forest grid with two states: empty place (Ground) and healthy tree.
forest_state_grid = np.random.choice(
    [EMPTY_PLACE, HEALTHY_TREE],
    size=(M, N),
    p=[0.3, 0.7],
).astype(np.uint8)


def fulfill_water(
    array: npt.NDArray,
    ellipse_start_x: int,
    ellipse_start_y: int,
    radius: float,
    c_radius: float,
    theta: float | int,
) -> npt.NDArray:
    from skimage.draw import ellipse

    # Drawing an ellipse shape within an array values (lake).
    rr, cc = ellipse(
        ellipse_start_x, ellipse_start_y, radius, c_radius, rotation=np.deg2rad(theta)
    )
    array[rr, cc] = WATER
    return array


# Draw water in shape of an ellipse in the center of the array.
RADIUS_ELLIPSE = 7
C_RADIUS_ELLIPSE = 10
ANGLE = 80
# Make a lake on the grid in the middle of an array.
forest_state_grid = fulfill_water(
    forest_state_grid,
    forest_state_grid.shape[0] // 2,
    forest_state_grid.shape[1] // 2,
    radius=RADIUS_ELLIPSE,
    theta=ANGLE,
    c_radius=C_RADIUS_ELLIPSE,
)

# Initialization of the fire.
healthy_trees = np.argwhere(forest_state_grid == HEALTHY_TREE)
IGNITE_INIT_CORD = healthy_trees[np.random.randint(healthy_trees.shape[0], size=1)][0]
forest_state_grid[IGNITE_INIT_CORD[0], IGNITE_INIT_CORD[1]] = BURNING_TREE

# Probability threshold of the fire adjacent trees; probability threshold of the self-ignition caused by e.g. lightning-struck.
IGNITE_PROB, SELF_IGNITE_PROB = 0.5, 0.001
# The number of generations after the tree will be restored.
TREE_REGROWTH_K = 20
# Fire direction change probability threshold.
FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH = 1
# Change of the fire direction every k generation.
FIRE_DIRECTION_K = 5

# Initialize the direction of the fire spread.
def get_wind_direction(neighbors: any) -> tuple[npt.NDArray, ...]:
    wind_direction_probabilities = np.array(
        [np.random.uniform() for _ in range(len(neighbors))]
    )
    wind_direction_indx = np.argmax(wind_direction_probabilities)
    wind_direction_probabilities[
        wind_direction_indx
    ] = FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH
    wind_direction_probabilities[
        wind_direction_probabilities != FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH
    ] = 0
    return wind_direction_probabilities, wind_direction_indx


wind_direction_indx = get_wind_direction(TREE_NEIGHBORS)[1]
WIND_DIRECTION_LABEL = DIRECTIONS[wind_direction_indx]


generation_to_regrowth_tree = defaultdict(list)

# Function simulates the fire within trees in the forest.x`
def cell2fire(
    forest_state_grid: npt.NDArray[np.uint8],
    ignite_tree_prob_threshold: float,
    self_ignition_prob_threshold: float,
    ignite_tree_wind_prob_threshold: float,
    tree_regrowth_after_k_gens: int,
    fire_direction_every_k_gen: int,
) -> npt.NDArray[np.uint8]:
    global wind_direction_indx
    global WIND_DIRECTION_LABEL

    future_forest_state = forest_state_grid.copy()
    generation = next(copy.copy(anim.frame_seq))

    # E, N, W, S - probability to spread the fire towards particular direction.
    # Rule 5. - Change of the direction of the fire based on the wind, change every n iters.
    if fire_direction_every_k_gen != 0 and generation % fire_direction_every_k_gen == 0:
        wind_direction_indx = get_wind_direction(TREE_NEIGHBORS)[1]
        WIND_DIRECTION_LABEL = DIRECTIONS[wind_direction_indx]

    for (x, y), cell_state in np.ndenumerate(forest_state_grid):
        # Rule 2. - The burning tree will be burnt in the next generation.
        if cell_state == BURNING_TREE:
            future_forest_state[x, y] = BURNT_TREE
            if tree_regrowth_after_k_gens != 0:
                generation_to_regrowth_tree[
                    generation + tree_regrowth_after_k_gens
                ].append((x, y))
        elif cell_state == HEALTHY_TREE:
            # Rule 1. - Tree will be ignited with probability `ignite_probability` if the adjacent trees are ignited.
            for idx, (nx, ny) in enumerate(TREE_NEIGHBORS):
                if (
                    0 < x + nx < future_forest_state.shape[0]
                    and 0 < y + ny < future_forest_state.shape[1]
                ):
                    tree_neighbor = forest_state_grid[x + nx, y + ny]
                    ignite_tree_prob = np.random.uniform()
                    if tree_neighbor == BURNING_TREE and ignite_tree_prob <= (
                        ignite_tree_prob_threshold
                        if wind_direction_indx != idx
                        else ignite_tree_wind_prob_threshold
                    ):
                        future_forest_state[x, y] = BURNING_TREE
                        break
            # Rule 3. - Self-ignition with certain probability.
            self_ignite_probability = np.random.uniform()
            if self_ignite_probability <= self_ignition_prob_threshold:
                future_forest_state[x, y] = BURNING_TREE

    # Update the state of the generation in which burnt tree will be the healthy one.
    while len(generation_to_regrowth_tree[generation]) != 0:
        x, y = generation_to_regrowth_tree[generation].pop()
        future_forest_state[x, y] = HEALTHY_TREE

    # Termination of the simulation if the all healthy tree were burnt.
    if np.all(animate.forest != BURNING_TREE):
        anim.event_source.stop()

    return future_forest_state


# Matplotlib settings for visualization of the forest fire.
cmap = colors.ListedColormap(["#362419", "#228B22", "#E25822", "#B2BEB5", "#42c2e5"])

# Figure size.
fig_size = (15, 10)
fig, ax = plt.subplots(figsize=fig_size)
ax.set_axis_off()
im = ax.imshow(forest_state_grid, cmap=cmap, vmin=EMPTY_PLACE, vmax=WATER)
colors = [im.cmap(im.norm(value)) for value in sorted(POSSIBLE_STATES)]
states = ["Ground", "Healthy tree", "Burning tree", "Burnt tree", "Water"]
patches = [
    patches.Patch(color=colors[i], label='State {d} - "{l}"'.format(l=states[i], d=i))
    for i in range(len(states))
]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


# Animate the fire of the forest.
def animate(iteration: int) -> None:
    plt.title(f"Generation: {iteration}, direction of the wind: {WIND_DIRECTION_LABEL}")
    im.set_data(animate.forest)
    animate.forest = cell2fire(
        animate.forest,
        ignite_tree_prob_threshold=IGNITE_PROB,
        self_ignition_prob_threshold=SELF_IGNITE_PROB,
        tree_regrowth_after_k_gens=TREE_REGROWTH_K,
        ignite_tree_wind_prob_threshold=FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH,
        fire_direction_every_k_gen=FIRE_DIRECTION_K,
    )


animate.forest = forest_state_grid
interval = 200
anim = animation.FuncAnimation(
    fig, animate, interval=interval, repeat=False, blit=False
)
plt.suptitle(
    f"Prob. thresh. of ignition neighbor: {IGNITE_PROB}; "
    f"Self-ignition prob. thresh: {SELF_IGNITE_PROB}; "
    f"K gens. to tree regrowth: {TREE_REGROWTH_K}; "
    f"Prob. of changing direction of the fire: {FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH}; "
    f"no. states: {NUM_STATES}; "
)
plt.show()
