from typing import *

# Provide indexing into a list that jumps around a lot
# ideal for selecting varied colors from a large linear color palette
def _palette_index(size: int) -> Iterator[int]:
    step = size
    current = size - 1
    used = set([])
    for i in range(size):
        used.add(current)
        yield current

        while current in used:
            current += step
            if current >= size:
                step //= 2
                current = 0
            if step == 0:
                break

    return