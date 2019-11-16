"""
"""

import random
from typing import *
from dataclasses import dataclass, field

# Parameters
population_size = 2
pheromone_default = 1
evaporation_rate = 0.9
hill_climb_iterations = 1
elitist_learning_rate = 0.2
elitist_selection_probability = 1
component_selection_heuristic = 1
component_selection_pheromone = 1
max_weight = 10


@dataclass
class Component:
    weight: int
    value: int
    pheromone: float = field(default=pheromone_default)

    def __str__(self):
        return f"{weight}kg, ${value}"


@dataclass
class Trail:
    components: List[Component] = field(default_factory=list)
    fitness: int = field(init=False)

    @property
    def total_weight(self) -> int:
        return sum([c.weight for c in self.components])

    @property
    def total_value(self) -> int:
        return sum([c.value for c in self.components])


def proportional_selection(components: List[Component]) -> List[Component]:
    """ """

    pass


components = [
    Component(5, 10),
    Component(4, 40),
    Component(6, 30),
    Component(3, 50),
]

best_trail_history = list()
best_trail = None
i = 0
while True:
    i += 1
    for _ in range(population_size):
        new_trail = Trail()
        while True:
            feasable_components = [
                c
                for c in components
                if c.weight + new_trail.total_weight <= max_weight
                and c not in new_trail.components
            ]

            if len(feasable_components) != 0:
                # TODO convert to Elitist component selection
                new_trail.components.append(random.choice(feasable_components))
            else:
                break

        new_trail.fitness = new_trail.total_value - (
            max_weight - new_trail.total_weight
        )

        

        if best_trail == None or new_trail.fitness > best_trail.fitness:
            best_trail = new_trail
            print(best_trail)
            best_trail_history.append(best_trail)

    for component in components:
        component.pheromone = (1 - evaporation_rate) * component.pheromone + (
            evaporation_rate * pheromone_default
        )

    for component in best_trail.components:
        component.pheromone = (1 - elitist_learning_rate) * component.pheromone + (
            elitist_learning_rate * best_trail.fitness
        )