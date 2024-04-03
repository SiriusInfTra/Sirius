from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import Optional
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Offline group')
parser.add_argument('--storage-nbytes-file', type=str, help='nbytes file')
parser.add_argument('--output', type=str, help='output file')
args = parser.parse_args()

# filename = 'nbytes_357309776.txt'
filename = args.storage_nbytes_file
with open(filename) as f:
    nbytes_ordered_list = np.array([np.int64(line) for line in f])
print(f'Max tensor {np.sort(nbytes_ordered_list)[-5:] / 1024 / 1024} MB')
print(f'Min tensor {np.sort(nbytes_ordered_list)[: 5]} B')
nbytes_ordered_cumsum = np.cumsum(nbytes_ordered_list)
GROUP_NBYTES_MIN  = 32 * 1024 * 1024
GROUP_NBYTES_MAX  = 128 * 1024 * 1024
GROUP_NBYTES_ALIGN = 16 * 1024 * 1024



def align(i: int | float):
    assert i % 1 == 0
    return (i + GROUP_NBYTES_ALIGN - 1) // GROUP_NBYTES_ALIGN * GROUP_NBYTES_ALIGN

@total_ordering
class Solution:
    def __init__(self, solution: Optional[Solution] = None, new_parti_pos: Optional[int] = None) -> None:
        assert (solution is None and new_parti_pos is None) or (solution is not None and new_parti_pos is not None)
        if solution is None and new_parti_pos is None:
            self.partitions_before = [0]
            self.fragment_nbytes = 0
            self.groups_size = []
        else:
            group_size = nbytes_ordered_cumsum[new_parti_pos - 1] \
                - (nbytes_ordered_cumsum[solution.partitions_before[-1] - 1] if solution.partitions_before[-1] > 0 else 0)
            self.partitions_before = [*solution.partitions_before, new_parti_pos]
            self.fragment_nbytes = solution.fragment_nbytes + (align(group_size) - group_size)
            self.groups_size = np.array([*solution.groups_size, group_size])

    def is_final(self) -> bool:
        return self.partitions_before[-1] == len(nbytes_ordered_list) 

    def __str__(self) -> str:
        return "{" + f"\n \
            parti={self.partitions_before},\n \
            frag={(self.fragment_nbytes / 1024 / 1024):.2f},\n \
            g_sz={['%.2f' % (sz / 1024 / 1024) for sz in self.groups_size]}\n \
        " + "}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return False
        return self.partitions_before == other.partitions_before
    
    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Solution)
        
        return (np.max(self.groups_size), -np.count_nonzero(np.logical_and(48 * 1024 * 1024 < self.groups_size, self.groups_size <= 64 * 1024 * 1024))) \
            < (np.max(other.groups_size), -np.count_nonzero(np.logical_and(48 * 1024 * 1024 < other.groups_size, other.groups_size <= 64 * 1024 * 1024)))
    
if np.max(nbytes_ordered_list) > GROUP_NBYTES_MAX:
    print(f'Max: {max(nbytes_ordered_list) / 1024 / 1024} greater than {GROUP_NBYTES_MAX / 1024 / 1024}')
    assert list(filter(lambda nbytes: nbytes > GROUP_NBYTES_MAX), nbytes_ordered_list) == 1
print(f'Len: {len(nbytes_ordered_list)}')
import math
import numpy as np
from collections import deque

final_solutions: list[Solution] = []
min_fragment = np.iinfo(np.int64).max

print(f'Ideal fragment nbytes: {(align(nbytes_ordered_cumsum[-1]) - nbytes_ordered_cumsum[-1]) / 1024 / 1024}')
q: deque[Solution] = deque()
q.appendleft(Solution())
iter_cnt = 0
cut_cnt = 0
while len(q) > 0:
    iter_cnt += 1
    solution = q.pop()
    if solution.is_final():
        if solution.fragment_nbytes < min_fragment:
            final_solutions = [solution]
            min_fragment = solution.fragment_nbytes
        elif solution.fragment_nbytes == min_fragment:
            final_solutions.append(solution)
        # if solution.partition_before == [0, 140, 268, 319]:
        #     print(solution, min_fragment)
        #     raise "a"
    else:
        for group_size in range(GROUP_NBYTES_MIN, GROUP_NBYTES_MAX + GROUP_NBYTES_ALIGN, GROUP_NBYTES_ALIGN):
            next_parti_pos = np.searchsorted(nbytes_ordered_cumsum, group_size + \
                (nbytes_ordered_cumsum[solution.partitions_before[-1] - 1] if solution.partitions_before[-1] > 0 else 0))
            if next_parti_pos == solution.partitions_before[-1]:
                continue
            new_solution = Solution(solution, next_parti_pos)
            if new_solution.fragment_nbytes > min_fragment:
                cut_cnt += 1
                continue
            q.appendleft(new_solution)
            if next_parti_pos == len(nbytes_ordered_cumsum):
                break
    if iter_cnt % 1000 == 0 or len(q) == 0:
        print(f'\rIterate: {iter_cnt:8d} Cut off: {cut_cnt:5d} Solution: {len(final_solutions):3d} \
              Min fragment: {(min_fragment / 1024 / 1024):.2f}', end='' if len(q) > 0 else '\n')

for i, solution in enumerate(sorted(final_solutions)):
    print(f'{i + 1}\t{solution}')
    if i + 1 >= 10:
        break

if args.output is not None:
    assert len(final_solutions) > 0
    solution = sorted(final_solutions)[0]
    with open(args.output, 'w') as f:
        for i in range(len(solution.partitions_before)):
            print(solution.partitions_before[i], file=f, 
                  end='\n' if i+1 != len(solution.partitions_before) else '')
    print(f'Output to {args.output}')
