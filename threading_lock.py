import threading
import time
# import logging
from concurrent.futures import ThreadPoolExecutor

def worker(name):
    print(f'Thread {name}: starting')
    time.sleep(3)
    print(f'Thread {name}: finishing')
    
    
class FakeDatabase:
    def __init__(self):
        self.value = 0
        
        # Ein Lock, um den Zugriff der Variable zu schützen
        # wird im unlocked Zustand initialisiert
        self.lock = threading.Lock()    
        
    def update(self, name):
        print(f'Thread {name}: starting update')
        
        # Die Zeit außerhalb des Locks warten alle Threads simultan
        time.sleep(2)
        
        # Mittels with kann aquired und released werden
        with self.lock:
            # Auf diesen Lock müssen die jeweiligen Threads warten, bis er
            # wieder freigegeben wird
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
        print(f'Thread {name}: finishing update')

if __name__ == '__main__':
    num_threads = 4
    now = time.time()
    
    # threads = list()
    
    # for index in range(5):
    #     print(f'Main: creat and start Thread {index}')
    #     thr = threading.Thread(target=worker, args=(index,))
    #     thr.start()
    #     threads.append(thr)
        
    # for idx, thr in enumerate(threads):
    #     print(f'Main {idx}: Joined')
    #     thr.join()
        
    # Executor -> führt aus einem Pool an Threads asynchrone Aufagben aus, 
    # wobei nur eine maximal Anzahl an Threads ausgeführt werden soll
    # with ThreadPoolExecutor(num_threads) as executor:
    #     # mapped die Liste auf die Funktion workers
    #     executor.map(worker, range(num_threads))
        
    my_database = FakeDatabase()
    with ThreadPoolExecutor(num_threads) as executor:
        for idx in range(8):
            # submit ruft eine callable funktion auf und übergibt named und 
            # unnamed argumente an die Funktion. Startet für jeden Aufruf
            # ein Thread
            executor.submit(my_database.update, idx)
    
    print(f'Testing Database Value: {my_database.value}')
    print(f'Main: Done {round(time.time() - now, 1)} s')
