from collections import deque

queue = deque()

def execute(task):
    print(f"Executing task: {task}")

while True:
    if queue: #cpu loads queue from the memory and do a branching instruction if queue is empty, so it wastes cpu cycles.
        task = queue.pop(0)
        execute(task)
