import time
from collections import deque

queue = deque()

def execute(task):
    print(f"Executing task: {task}")

while True:
    if queue:
        task = queue.pop(0)
        execute(task)
    else:
        time.sleep(0.1) #cpu will be idle (switch to kernel space and thread will be blocked) for 0.1 seconds, so it will not waste cpu cycles.
