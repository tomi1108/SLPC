import time
import threading

thread_flag = False
thread_size = 5

class myThread(threading.Thread):
    def __init__(self):
        super(myThread, self).__init__()
    
    def run(self):
        for i in range(0, 5):
            time.sleep(1)
            print("count", i)

class noThread():
    def __init__(self):
        pass # passとは、何もしないという意味

    def run(self):
        for i in range(0, 5):
            time.sleep(1)
            print("count", i)

start_time = time.time()
if __name__ == "__main__":
    if thread_flag:
         threads = []
         for i in range(thread_size):
             thread = myThread()
             threads.append(thread)
        
         for i in range(thread_size):
             threads[i].start()
        
         for i in range(thread_size):
             threads[i].join()
    else:
        noThreads = []
        for i in range(thread_size):
            noth = noThread()
            noThreads.append(noth)
        
        for i in range(thread_size):
            noThreads[i].run()

end_time = time.time()
print("time", end_time - start_time)