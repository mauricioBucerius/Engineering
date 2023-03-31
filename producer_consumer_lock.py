import random
import threading
from concurrent.futures import ThreadPoolExecutor

# Sentianl Value, wird benutzt, um das Ende anzuzeigen -> Dummy Objekt
SENTINEL = object() 

def producer(pipeline):
    # producer, der aus dem Fake Netzwerk liest und die Nachricht in die 
    # Pipeline schreibt
    for index in range(10):
        msg = random.randint(1, 101)    # Message aus dem Fake Netzwerk
        print('Producer got message')
        pipeline.set_message(msg, 'Producer')
        
    pipeline.set_message(SENTINEL, 'Producer')
    
def consumer(pipeline):
    # Consumer, der aus der Pipeline liest und den Wert in eine Fake Datenbank
    # einspeichert, bis er ein SENTINEL erhält und den Thread beendet
    msg = 0
    while msg is not SENTINEL:
        # list die Nachricht aus der Pipeline
        msg = pipeline.get_message("Consumer")
        if msg is not SENTINEL:
            print(f'Consumer storing message: {msg}')

class Pipeline:
    def __init__(self):
        self.msg = 0    # speichert die Nachricht, die übertragen werden soll
        
        # producer_lock beschränkt den Zugriff durch den Producer Thread
        self.producer_lock = threading.Lock()
        
        # consumer_lock beschränkt den Zugriff durch den Consumer Thread
        self.consumer_lock = threading.Lock()
        
        self.consumer_lock.acquire()    # Das ist der State mit dem man startet
        
    def get_message(self, name):
        # Nachricht wird von Consumer aus der Pipeline gelesen
        print(f'{name}: about to aquire lock')
        self.consumer_lock.acquire()
        print(f'{name}: aquired lock')
        message = self.msg
        print(f'{name}: about to relrease lock')
        self.consumer_lock.release()
        print(f'{name}: relreased lock')
        return message
    
    def set_message(self, msg, name):
        # Nachrichten wird von Producer in die Pipeline geschrieben
        print(f'{name}: about to aquire lock')
        self.producer_lock.acquire()
        print(f'{name}: aquired lock')
        self.msg = msg
        print(f'{name}: about to release lock')
        self.producer_lock.release()
        print(f'{name}: released lock')
        
if __name__ == '__main__':
    num_threads = 2
    
    pipeline = Pipeline()
    
    with ThreadPoolExecutor(num_threads) as executor:
        executor.submit(producer, pipeline) # Schreibt aus Netzwerk in Pipeline
        executor.submit(consumer, pipeline) # List aus Pipeline in Database
