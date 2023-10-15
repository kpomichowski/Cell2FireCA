# Cell2Fire

Cell2Fire is a cellular automaton simulation that imitates the spread of fire in a forest.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Examples](#examples)

## Installation

Before installation necessary dependencies to run this project, create Python virtual environment, in the terminal go the directory of the project:

`cd path/to/cell2fire/project`

Then create the Virtual Environment:

`python -m venv [env]`

If you are using Anaconda or Mamba, issue following command:

`conda create -n [env]` or `mamba create -n [env]`

## Windows case

In the case of Windows, just activate the Virutal Environment using:

`[env]\Scripts\activate`

## Linux case

In the case of standalone version of virtual environment using Python, you need to activate created environment:

`. ./[env]/bin/activate` or `source [env]/bin/activate`

In the case of `conda` or `mamba`:

`conda activate [env]` or  `mamba activate [env]`

`[env]` is the name of your created Virtual Environment.

When you notice that Your environment has been activated, then install necessary libraries and dependencies to run this cellular automata:

`pip install -r requirements.txt`

## Usage

This project is a cellular automaton simulating a forest fire. The cellular automata contains following states:
- tree,
- firing tree,
- burnt tree,
- water,
- ground (empty place).

The evolution of states is governed by the following rules:
- A tree will catch fire with a probability $p$ if one of its neighbors is burning,
- A burning tree at time $t+1$ will be consumed by the fire,
- Spontaneous ignition of a tree occurs with a probability $p$,
- Water acts as a barrier to fire,
- The fire spreads depending on the direction of the wind, which changes every k iterations;

The simulation will terminate if the ignition disappears. The ignition in the forest is initialized in a random location.

The parameters in the simulation:
- `IGNITE_PROB` — parameter $p$ which is responsible for ingiting a tree if in the neighboorhood of that tree is firing tree,
- `SELF_IGNITE_PROB`  — parameter which is responsible for a chance of self-igniting a tree,
- `TREE_REGROWTH_K` — parameter which is responsible for regenerating burnt tree after $k$ generations,
- `FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH` — parameter (probability threshold) for change the direction of the wind,
- `FIRE_DIRECTION` — parameter which is responsible for change the direction of the fire every $k$ generations.

The neighboorhood in this project was implemented based on the [Von Neumann Neighboorhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood).

## Dependencies

Project was built upon following librariers:
* matplotlib,
* numpy,
* scikit-image.

## Examples

Here you can find an example of the Ceullar Automata with following parameters:
- `IGNITE_PROB` = 0.5,
- `SELF_IGNITE_PROB` = 0.001,
- `TREE_REGROWTH_K` = 20,
- `FIRE_DIRECTION_CHANGE_PROBABILITY_THRESH` = 1,
- `FIRE_DIRECTION_K` = 5;
- 
![Animation](https://github.com/kpomichowski/Cell2FireCA/blob/master/output.gif)

