#SAM_Nominal_Uncategorized
"""
Created on Thu Mar 12 2020

@author: willamannering
"""
import numpy as np
import random

class SAM_Nominal_Uncategorized:
    
    def __init__(self, ListLength, t=2, r=4, sam_a = .08, sam_b = .08, sam_c = .08, 
                 sam_d = .02, sam_e = 0.7, sam_f = 0.7, sam_g = 0.7, Kmax = 30, Lmax = 3):

        #Parameter settings based on parameter estimation in first paper:
        #sam_e = 0.7, sam_f = 0.72, sam_g = 0.7

        ''' ListLength = number of items in studylist, 
        t = presentation time per word
        r = short term memory buffer
        sam_a = multiplier for context to word association
        sam_b = mulitplier for word cue to other word trace association
        sam_c = multiplier for word cue to same word trace association
        sam_d = residual strength of association for words that never appear in buffer together
        sam_e = incrementing parameter for context to word association
        sam_f = incrementing parameter for word to other word association
        sam_g = incrementing parameter for word to itself association
        Kmax = maximum number of retrieval failures before search process is stopped
        Lmax = max number of retrieval attempts using word cues instead of context
        '''
        


        self.ListLength = ListLength
        self.t = t
        self.r = r
        self.sam_a = sam_a
        self.sam_b = sam_b
        self.sam_c = sam_c
        self.sam_d = sam_d
        self.sam_e = sam_e
        self.sam_f = sam_f
        self.sam_g = sam_g
        self.Kmax = Kmax
        self.Lmax = Lmax
        self.K = 0
        self.L = 0
        self.context_assoc, self.word_assoc = self.encodeitems()
        
    #method for encoding items
    def encodeitems(self):
        import itertools
        
        buffer = []
        present_order = list(range(self.ListLength))
        
        random.shuffle(present_order) #randomize presentation order
        
        #initialize context association and word-word association matrices
        context_assoc = np.zeros((1, self.ListLength))
        word_assoc = np.zeros((self.ListLength, self.ListLength))
        
        loops_inbuffer = [0]*self.ListLength #what loops were all items in the buffer?
        
        for i in range(self.ListLength):
            if i >= self.r: #if buffer is full
                buffer[random.randint(0, self.r-1)] = present_order[i] #randomly replace item in buffer with next item
                
            else: #if buffer isn't full
                buffer.append(present_order[i]) #add next item to buffer
                
            
            to_update = list(itertools.permutations(buffer,2))
            
            for i in range(int(self.t)): #presentation time, if t is larger words stay in buffer updating associations for longer
                for pair in to_update:
                    word_assoc[pair[0]][pair[1]] = word_assoc[pair[0]][pair[1]]+ (self.sam_b)
                for j in range(len(buffer)):
                    word_assoc[buffer[j]][buffer[j]] = word_assoc[buffer[j]][buffer[j]] + (self.sam_c)
                for j in buffer:
                    loops_inbuffer[j] += 1
            
            word_assoc[word_assoc == 0] = self.sam_d
            
            
        #create context association matrix     
        context_assoc = np.array([self.sam_a*loops for loops in loops_inbuffer])
       
    
        return context_assoc, word_assoc
    
    #method for updating association matrix
    def update_assoc(self, sampledTrace,  wordcue = -1):
    
        if wordcue != -1: #if word was used as cue, update strengths between cue and retrieved image
            
            self.context_assoc[sampledTrace] = self.context_assoc[sampledTrace] + self.sam_e
            self.word_assoc[sampledTrace][sampledTrace] = self.word_assoc[sampledTrace][sampledTrace] + self.sam_g
            self.word_assoc[wordcue][sampledTrace] = self.word_assoc[wordcue][sampledTrace] + self.sam_f 
            self.word_assoc[sampledTrace][wordcue] = self.word_assoc[sampledTrace][wordcue] + self.sam_f 
        
        else: #if only context was used, update association between image to context and image to itself
            self.context_assoc[sampledTrace] = self.context_assoc[sampledTrace] + self.sam_e
            self.word_assoc[sampledTrace][sampledTrace] = self.word_assoc[sampledTrace][sampledTrace] + self.sam_g
        
    #free recall process 
    def free_recall(self):

        #begin free recall process
        alreadySaid = [False] * self.ListLength #at this point no words have been said from the list
        
        response = [] #empty list of free recall responses
         #number of retrieval failures, recall continues until retrival failures is greater than Kmax

        while(self.K < self.Kmax):
            self.L = 0
    
            probSamp = [ci/np.sum(self.context_assoc) for ci in list(self.context_assoc)]
    
            sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0] #begin free recall by using context as a search cue
        
            if (alreadySaid[sampledTrace]):
                self.K += 1
                    
            else: #otherwise, if a new trace was sampled
                    
                probRecover = 1-np.exp(-self.context_assoc[sampledTrace])
               
                if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                    
                    self.update_assoc(sampledTrace)
                    
                    alreadySaid[sampledTrace] = True #set this trace to already said
                    
                    response.append(sampledTrace) #record this trace as a successful response
                    
                    #if trace was recovered, use this word as cue to recover more traces
                    while(self.L < self.Lmax):
                        
                        previous_sample = sampledTrace

                        probSamp = (self.context_assoc*self.word_assoc[sampledTrace])/sum(self.context_assoc*self.word_assoc[sampledTrace])
                        
                        #randomly choose a trace using calculated probabilities
                        sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0]
                        
                        
                        if (alreadySaid[sampledTrace]): #if sampledTrace already said, count this as retrieval failure and start again
                            self.K += 1
                            self.L += 1                  
                            sampledTrace = previous_sample
                            
                        else: #otherwise, if a new trace was sampled
                            
                            probRecover = 1-np.exp((-self.context_assoc[sampledTrace])-self.word_assoc[previous_sample][sampledTrace])
                            
                            if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                                self.update_assoc(sampledTrace, wordcue = previous_sample)

                                alreadySaid[sampledTrace] = True #set this trace to already said
                                
                                response.append(sampledTrace) #record this trace as a successful response
                                    
                                self.L = 0 #set L to zero and use newly recovered trace as cue
        
                            else:
                                self.L+=1
                                sampledTrace = previous_sample

                else:
                    self.K += 1
    
        return response
