import numpy as np

class Optimise:
    
    
    def greedy(self,f,iterations,start):
        
        steps = [0.0001 , 0.001,0.01,0.1,1,10,100,1000,10000]
        directions = [1,0,-1]
        pars = [0,1]
        current = start.copy()
        best  = 10**10
        for x in range(0, iterations):
            bestd = -1
            besth=-1
            bestp = -1
            for h in steps:
                for d in directions:
                    for p in pars:
                        test = current.copy()
                        test[p]+= d * h
                    
                        if f(test) < best:
                            best  = f(test)
                            besth = h
                            bestd = d
                            bestp  = p
            
            
            if bestp!=-1:
                current[bestp]+=bestd*besth
        
        
        return current
            
   
    def bfgs(self,f,iterations,start):
        
        current = np.array(start.copy())
        steps = [0.0001 , 0.001,0.01,0.1,1,10,100,1000,10000]
        best = 10**10
        besth = -1 
        binv = np.eye(len(current))
        
        for x in range(0, iterations):
            
            p  =  np.dot(binv,self.grad(f,current))
            
            for h in steps:
                test  = current - p * h
                if f(test) < best:
                    best  = f(test)
                    besth = h
            
            
            s = np.matrix(besth * p )
            y = np.matrix( self.grad(f,current + besth*p) - self.grad(f,current))
            b1 = np.dot(y,np.transpose(s))[0,0]
            b2  = np.dot(np.dot(s,binv),np.transpose(s))[0][0]
            
            yy  = np.dot(np.transpose(y),y)/b1
            bb  = np.dot(np.dot(binv,np.transpose(s)),np.dot(s,binv))/b2
            
            print(yy,bb)
            binv =np.array( binv+yy-bb)
            
            current -= np.dot(binv,self.grad(f,current))
             
            
            print (current)
            
            
        
        return current
    
    def newt_mod(self,f,iterations,start):
        
        current = np.array(start.copy())
        steps = [0.0001 , 0.001,0.01,0.1,1,10,100,1000,10000]
        best = 10**10
        besth = -1 
        bk = np.eye(len(current))
        
        for x in range(0, iterations):
            for h in steps:
                bmod = bk.copy() + h * np.eye(len(current))
                
                binv =  np.linalg.pinv(bmod)
                test = current - np.dot(binv,self.grad(f,current))
                
                if f(test) < best:
                    best  = f(test)
                    besth = h
                
            
            
            binv =  np.linalg.pinv(bk+besth*np.eye(len(current)))
            current-= np.dot(binv,self.grad(f,current))
        
        return current
    
    def lm(self,f,iterations,start):
   
        current = np.array(start.copy())
        steps = [0.0001 , 0.001,0.01,0.1,1,10,100,1000,10000]
        best = 10**10
        besth = -1 
        for x in range(0, iterations):
            for h in steps:
                jtj = np.dot(np.transpose(s.jacobian(g,current)),s.jacobian(g,current) )
                jtji =  np.linalg.pinv(jtj + h* np.eye(len(current)))
                test = current - np.dot(jtji,self.grad(f,current))
                
                if f(test) < best:
                    best  = f(test)
                    besth = h
                
            
            jtjBest = np.dot(np.transpose(s.jacobian(g,current)),s.jacobian(g,current) )
            jtjiBest =  np.linalg.pinv(jtjBest + besth* np.eye(len(current)))
             
            current-= np.dot(jtjiBest,self.grad(f,current))
        
        return current
    
        
    
    def gd(self,f,iterations,start):
        
        current = np.array(start.copy())
        steps = [0.0001 , 0.001,0.01,0.1,1,10,100,1000,10000]
        best = 10**10
        besth = -1 
        for x in range(0, iterations):
            for h in steps:
                
                test = current - h * self.grad(f,current)
                
                if f(test) < best:
                    best  = f(test)
                    besth = h
                
                
            current-= besth*self.grad(f,current)
        
        return current
    
    
    
    
    
    def binsearch(self,vals,y,iterations):
        sorted = self.qs(vals,0)
    
        lower = 0
        upper = len(sorted)
        
        for x in range(0,iterations):
         
            if y < sorted[int((upper+lower)/2)]:
                upper =  int((upper+lower)/2)
            else:
                lower = int((upper+lower)/2)

        return upper
                          
    
    def newton (self,f,iterations,start):
        h =start
        for x in range(0, iterations):
            h-= f(h)/self.deriv(f,h)
        return h
    
    
                                      
    def jacobian(self, f, pars):
        #normally this is n * pars, in this case it's 1 * pars so grad transpose
        
        output = np.zeros((1,len(pars)))
        dz =0.00001
        for x in range(0,len(pars)):
            fz  = f(pars)
            pars[x]+=dz
            fzdz = f(pars)
            pars[x]-=dz
            output[0,x] = (fzdz - fz)/dz
        return output
                                      
    
                                      
    def grad(self,f,pars):
        
        output = np.zeros(len(pars))
        dz =0.00001
        for x in range(0,len(pars)):
            fz  = f(pars)
            pars[x]+=dz
            fzdz = f(pars)
            pars[x]-=dz
            output[x] = (fzdz - fz)/dz
        return output
 

    def deriv(self,f,x):
        dx=0.0001
        return (f(x+dx)-f(x))/dx
    
    
    
    def bisect(self,f, iterations):
        start = 0
        startrange = 1000
        r =  startrange
        for x in range(0, iterations):
            
            f1 = f(start + r)
            
            if f1  > 0:
                r*=0.5
            else:
                
                start+=r
                r*=2
                
        return start+r
    
    def qs(self,vals,n):
        left = []
        right = []

        pivot = vals[n]
        for x in vals:
            if x < pivot:
                left.append(x)
            else:
                right.append(x)

        l=[]
        r=[]
        if len(left) > 1:
             l = self.qs(left,n)

        if len(right) > 1:
            r = self.qs(right[1:],n)


        return l + [pivot] +  r 

    


    
def f(x):
    return (x-5)**2

def g(pars):
    x = pars[0]
    y=pars[1]
    return (x-2)**2 + (y-1.23)**2




s  = Optimise()
s.qs([4,2,88,10,3,19,26],0)
#s.binsearch([4,2,88,10,3,19,26],26,50)
#s.bisect(f,50)
#s.newton(f,50)
s.gd(g,50,[1.0,5.0]),s.lm(g,30,[1.0,5.0]),s.greedy(g,30,[1.0,5.0])             
