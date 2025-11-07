from task1 import findpricegappair

def runtests():
    print("Running Task1 tests\n")

    print("1. Basic example")
    nums = [4, 1, 6, 3, 8]
    k = 2
    print(findpricegappair(nums, k))  

    print("2. Multiple duplicates pick lexicographically smallest")
    nums = [5, 5, 5]
    k = 0
    print(findpricegappair(nums, k)) 

    print("3. Negatives and positives")
    nums = [-3, 7, 1, -1, 4]
    k = 2
    print(findpricegappair(nums, k))  

    
    print("4. No solution")
    nums = [10, 20, 30]
    k = 25
    print(findpricegappair(nums, k))  

    print("5. Edge case repeated numbers but different indices")
    nums = [1, 2, 1, 2]
    k = 1
    print(findpricegappair(nums, k))  

if __name__ == "__main__":
    runtests()
