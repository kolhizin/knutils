"""
Goal of this module is to provide class for queing jobs from queue and saving that job result in another queue.
"""

class InQlist:
    def __init__(self, lst):
        self.__data = lst
    
    def empty(self):
        return self.__data is None or len(self.__data) == 0 
        
    def pop(self, default = None):
        if self.empty():
            return None
        res = self.__data[0]
        self.__data = self.__data[1:]
        return res
    
class OutQlist:
    def __init__(self):
        self.__data = []
        
    def push(self, x):
        self.__data.append(x)
        
    def close(self):
        pass
    
class LogQprint:
    def __init__(self):
        pass 
    
    def log_start(self):
        print('start')
        
    def log_pop(self):
        print('q-pop')
    
    def log_input(self, x):
        print('in: {}'.format(x))
        
    def log_transform_error(self, x, e):
        print('err: {0}, {1}'.format(x, e))
        
    def log_output(self, x):
        print('out: {}'.format(x))
        
    def log_push(self, x):
        print('push: {}'.format(x))
        
    def log_close(self):
        print('close')
        
    def close(self):
        pass
    
class QExec:
    def __init__(self, qin, qout, qlog):
        self.__qin = qin
        self.__qout = qout
        self.__qlog = qlog
        
    def transform_one(self, tfunc):
        self.__qlog.log_pop()
        x = self.__qin.pop()
        
        if x is None:
            return
        
        self.__qlog.log_input(x)
        try:
            y = tfunc(x)
        except Exception as e:
            self.__qlog.log_transform_error(x, e)
        except:
            self.__qlog.log_transform_error(x, 'unknown')
        else:
            self.__qlog.log_output(y)
            self.__qout.push(y)
            self.__qlog.log_push(y)
        
    def transform_all(self, tfunc):
        self.__qlog.log_start()
        
        while not self.__qin.empty():
            self.transform_one(tfunc)
        
        self.__qlog.log_close()