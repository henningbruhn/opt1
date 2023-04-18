#################### The simplex algorithm ###############################

# conceptual implementation, not suitable for performance
# only for educational purpose

import numpy as np
import math

class State:
    """
    simple class that keeps track of what's happening in each simplex step
    the following parameters are stored:
    x,J,y_J,i,j*,w,lambda,the new vertex x, the current objective value, whether the OPT was reached
    """
    def __init__(self,x=None,J=None,y_J=None,i=None,jstar=None,w=None,opt_status="running",lmbda=None,new_x=None,objective=None):
        self.x=x.flatten()
        self.J=J.copy() if J is not None else None # a little bit of python magic to cope with J==None
        self.y_J=y_J.flatten() if y_J is not None else None 
        self.i=i
        self.jstar=jstar
        self.w=w.flatten() if w is not None else None
        self.opt_status=opt_status
        self.lmbda=lmbda
        self.new_x=new_x.flatten() if new_x is not None else None
        self.objective=objective
    
    def __repr__(self):
        repr=""
        # only report those values that are not None
        for key,value in vars(self).items():
            if value is not None:
                repr+="{}: {}\n".format(key,value)
        return repr+"\n"

def unit_vec(i,n):
    """
    return unit vector e_i of length n
    """
    vec=np.zeros(n)
    vec[i]=1
    return vec.reshape(-1,1)

class Simplex:
    def __init__(self,A,b,c,x0,max_iter=100,eps=1E-8,verbose=True):
        """
        solve max c^Tx s.t. Ax<=b
        A: matrix of conditions
        b: right-hand side
        c: vector of objective function
        x0: start vertex
        max_iter: maximum number of simplex steps (default=100)
        eps: numerical precision (default=10^-8)
        verbose: True/False, whether to report optimisation result (default=True)
        when finished, the following attributes are set
        simplex.x: the optimal solution (if exists)
        simplex.opt_status: string "OPT found" / "unbounded"
        simplex.record: a list containing a State object for each simplex step
        """
        self.A=np.array(A) # make sure it's a numpy array
        self.b=np.array(b).reshape(-1,1) # make sure it's a numpy array column vector
        self.c=np.array(c).reshape(-1,1) # make sure it's a numpy array  column vector
        self.x=np.array(x0).reshape(-1,1) # make sure it's a numpy array  column vector
        m,n=self.A.shape
        self.m=m
        self.n=n
        self.eps=eps
        self.record=[]
        self.done=False
        self.max_iter=max_iter
        self.check_start_vertex()
        self.init_representation()
        self.run()
        if verbose:
            self.report_result()
        
    def check_start_vertex(self):
        """
        check whether x0 is indeed feasible
        """
        # check whether A*x0<=b because of numerical imprecisions allow up to eps deviation
        if not np.all(self.A@self.x-self.b <=self.eps): 
            raise Exception("start parameter x0 not feasible!")
        
    def current_objective(self):
        """
        objective value of current solution simplex.x
        """
        return np.sum(self.x.T@self.c)
        
    def run(self):
        """
        run the actual algorithm
        """
        for _ in range(self.max_iter):
            if self.done:
                break
            self.simplex_step()
        if not self.done:
            self.opt_status="max iteration reached"
        
    def init_representation(self):
        """
        compute representing index set J for start solution x0
        """
        # collect the indices of tight inequalities
        # allow up to eps deviation due to numerical imprecisions
        self.J=[j for j,a in enumerate(self.A) if abs(a@self.x-self.b[j])<self.eps]
        # check whether A_J has indeed rank n
        m,n=self.A.shape
        if np.linalg.matrix_rank(self.A[self.J])<n:
            raise Exception("start parameter x0 not a vertex!")
        self.reduce_index_set() # throw away inequalities in J if |J|>n
        
    def reduce_by_one(self):
        m,n=self.A.shape
        for j in range(len(self.J)):
            JJ=self.J.copy()
            del JJ[j]
            if np.linalg.matrix_rank(self.A[JJ])>=n:
                break
        del self.J[j]
        
    def reduce_index_set(self):
        """
        throw away inequalities in simplex.J if |J|>n
        while making sure that still rank(A_J)=n
        """
        m,n=self.A.shape
        while len(self.J)>n:
            self.reduce_by_one()
        
    def simplex_step(self):
        """
        a single step of the simplex algorithm
        """
        ### 1: reduce matrix to the rows in J
        A_J=self.A[self.J]
        ### 2: compute y_J
        self.y_J=np.linalg.solve(A_J.T,self.c)
        obj=self.current_objective()                              # for record keeping
        state=State(x=self.x,J=self.J,y_J=self.y_J,objective=obj) # for record keeping
        self.record.append(state)                                 # for record keeping
        ### 3: pick index i with y_i<0
        index_i_found=False
        for rel_i,yy in enumerate(self.y_J):
            if yy<-self.eps: # is y_i<0? up to eps due to numerical imprecisions
                index_i_found=True
                break
        ### 4: check whether optimum found
        if not index_i_found:
            self.done=True
            self.opt_status="OPT found"
            state.opt_status="OPT found"                          # for record keeping
            return
        state.i=self.J[rel_i]                                     # for record keeping
        ### 5: compute w, direction in which objective value can be improved
        w=np.linalg.solve(A_J,-unit_vec(rel_i,len(self.J)))
        state.w=w.flatten()                                       # for record keeping
        ### 6: check whether A*w<=0, ie whether LP is unbounded
        if np.all(self.A@w<=self.eps):
            self.done=True
            self.opt_status="unbounded"
            state.opt_status="unbounded"                          # for record keeping
            return
        ### 7: compute lambda and j*, the index to add to J
        lambdas=[(sum((self.b[j]-sum(self.A[j]@self.x))/sum(self.A[j]@w)),j) for j in range(len(self.A)) if sum(self.A[j]@w)>self.eps]
        lmbda,jstar=min(lambdas,key=lambda l:l[0])
        state.jstar=jstar                                         # for record keeping
        state.lmbda=lmbda                                         # for record keeping
        ### 8: update J by deleting i and adding j*
        self.J[rel_i]=jstar
        self.J.sort()
        ### 9: compute new vertex x defined by J
        self.x=np.linalg.solve(self.A[self.J],self.b[self.J])
        state.new_x=self.x.flatten()                              # for record keeping
        
    def report_result(self):
        steps=len(self.record)
        if self.opt_status=="unbounded":
            print("LP unbounded")
        if self.opt_status=="OPT found":
            print("OPT found")
            print(" at x={}".format(self.x.flatten()))
            print(" OPT value={}".format(self.current_objective()))
        if self.opt_status=="max iteration reached":
            print("algorithm stopped as max number of iterations was reached")
        print("simplex took {} steps".format(steps))
 
###################### some sample linear programmes ############################     

def add_nonneg(A,b):
    """
    add non-negativity constraints x>=0 to system Ax<=b
    """
    m,n=A.shape
    b=np.vstack([b,np.zeros(n).reshape(-1,1)])
    A=np.vstack([A,-np.eye(n)])
    return A,b

def twisted_cube(n=3):
    c=np.array([2**i for i in range(n-1,-1,-1)]).reshape(-1,1)
    b=np.array([5**i for i in range(1,n+1)]).reshape(-1,1)
    A=2*np.array([[int(2**(i-j)) for j in range(n)] for i in range(n)])
    for i in range(n):
        A[i,i]=1
    A,b=add_nonneg(A,b)
    R=np.array([[i+n,i] for i in range(n)]).flatten()
    A=A[R].copy()
    b=b[R].copy()
    x0=np.zeros(n).reshape(-1,1)
    return A,b,c,x0

def dodecahedron():
    phi=0.5*(math.sqrt(5)+1)
    generators=[[0,(-1)**i,(-1)**round(0.4*i)*phi] for i in range(4)]
    ineqs=[]
    for generator in generators:
        for _ in range(3):
            ineqs.append(generator.copy())
            generator.append(generator.pop(0))
    A=np.vstack(ineqs)
    b=np.ones(len(A)).reshape(-1,1)
    x0=np.array([-1/phi**3,0,1-phi])
    c=np.array([0,0,1]).reshape(-1,1)
    return A,b,c,x0

def octahedron():
    A=np.array([[i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]])
    b=np.ones(8).reshape(-1,1)
    c=np.array([0,0,1]).reshape(-1,1)
    x0=np.array([0,0,-1]).reshape(-1,1)
    return A,b,c,x0

def flat():
    A=np.array([[2,-1,0],[1,0,0],[0,1,0],[-1,1,0],[-1,0,0],[0,-1,0],[1,1,-1],[-1,-1,1]])
    b=np.array([2,3,5,4,0,0,0,0]).reshape(-1,1)
    c=np.ones(3).reshape(-1,1)
    x0=np.zeros(3).reshape(-1,1)
    return A,b,c,x0

def sample_poly():
    cube_A,cube_b=add_nonneg(np.eye(3),4*np.ones(3).reshape(-1,1))
    more_A=[[1,2,-1],[1.2,-3.2,1.6],[-1,-1,1],[-0.25,0,1]]
    more_b=np.array([2,8,1.5,3]).reshape(-1,1)
    A=np.vstack([cube_A,more_A])
    b=np.vstack([cube_b,more_b])
    c=np.array([0,0,1]).reshape(-1,1)
    x0=np.zeros(3).reshape(-1,1)
    return A,b,c,x0
