import numpy as np

class ExperienceReplay():
    def __init__(self,maxlen):
        self.maxlen = maxlen
        self.memory = np.zeros([maxlen,10])
        self.counter = 0

    def add(self,current):
        self.memory[self.counter%self.maxlen] = np.hstack(current)
        self.counter+=1

    def get_batch(self,batch_size):
        sample_index = np.random.choice(self.maxlen,batch_size)
        return self.memory[sample_index,:]

    def __len__(self):
        return min(self.counter,self.maxlen)