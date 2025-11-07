from typing import List, Tuple, Optional
def findpricegappair(nums: List[int], k: int) -> Optional[Tuple[int, int]]:
    if k < 0:
        return None
    valuetoindex = {}
    bestpair = None
    for j, num in enumerate(nums):
        for target in (num - k, num + k):
            if target in valuetoindex:
                i = valuetoindex[target]
                if i < j:
                    if bestpair is None or (i, j) < bestpair:
                        bestpair = (i, j)
        if num not in valuetoindex:
            valuetoindex[num] = j
    return bestpair
