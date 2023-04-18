
#import cdd # pycddlib
import numpy as np
import plotly.graph_objects as go
import math
import ipywidgets as widgets
import simplex

###################### compute vertices and edges of polyhedron ##################

class Polyhedron:
    def __init__(self,A,b,enclose_factor=1000,eps=1E-8):
        A=np.array(A)
        b=np.array(b).reshape(-1,1)
        self.A=A
        self.b=b
        self.num_ineqs,_=self.A.shape
        self.eps=eps
        if enclose_factor>0:
            upper_bounds_A=np.eye(3)
            upper_bounds_b=enclose_factor*(np.ones(3).reshape(-1,1))
            lower_bounds_A=-np.eye(3)
            lower_bounds_b=enclose_factor*(np.ones(3).reshape(-1,1))
            self.A=np.vstack([A,upper_bounds_A,lower_bounds_A])
            self.b=np.vstack([b,upper_bounds_b,lower_bounds_b])
        self.m,self.n=self.A.shape
        self._get_vertices()
        self._get_edges()
        self._set_up_face_vxs()
        
    def get_range(self):
        """returns a list [xrange,yrange,zrange]"""
        return np.array([np.min(self.vertices,axis=0),np.max(self.vertices,axis=0)]).T
        
    def is_feasible(self,x):
        return np.all(self.A@x.reshape(-1,1)-self.b.reshape(-1,1) <=self.eps)

    def same_vx(self,x,y):
        return np.all(np.abs(x-y)<=self.eps)

    def _is_new_vx(self,x):
        for v in self.vertices:
            if self.same_vx(v,x):
                return False
        return True

    def _get_vertices(self):
        self.vertices=[]
        for i in range(self.m):
            for j in range(i+1,self.m):
                for k in range(j+1,self.m):
                    AA=self.A[[i,j,k]]
                    bb=self.b[[i,j,k]]
                    try:
                        x=np.linalg.solve(AA,bb).flatten()
                        if self.is_feasible(x) and self._is_new_vx(x):
                            self.vertices.append(x.flatten())
                    except np.linalg.LinAlgError:
                        pass
        self.vertices=np.array(self.vertices)
        
    def is_in_face(self,FA,Fb,x):
        return np.all(np.abs( FA@x.reshape(-1,1)-Fb.reshape(-1,1)) <=self.eps)
    
    def _is_new_edge(self,edge):
        u,v=edge
        for e in self.edges:
            if self.same_vx(u,e[0]) and self.same_vx(v,e[1]):
                return False
            if self.same_vx(u,e[1]) and self.same_vx(v,e[0]):
                return False
        return True

    def _get_edges(self):
        self.edges=[]
        for i in range(self.m):
            for j in range(i+1,self.m):
                AA=self.A[[i,j]]
                bb=self.b[[i,j]]
                if np.linalg.matrix_rank(AA)>=2:
                    edge=[]
                    for v in self.vertices:
                        if self.is_in_face(AA,bb,v):
                            edge.append(v)
                    if len(edge)==2 and self._is_new_edge(edge):        
                        self.edges.append(np.array(edge))
                        
    def _find_face_vxs(self,face_num):
        FA=self.A[[face_num]]
        Fb=self.b[[face_num]]
        face_vxs=[vx for vx in self.vertices if self.is_in_face(FA,Fb,vx)]
        return np.array(face_vxs)
    
    def _set_up_face_vxs(self):
        self.face_vxs=[self._find_face_vxs(face_num) for face_num in range(self.m)]
                

################### visualisation code #######################################

def perturb(vxs):
    """
    Mesh3d doesn't seem to work properly if vxs are in same plane
    for this reason we jitter the vxs a little bit
    """
    n=len(vxs)
    return vxs+np.random.random(size=(n,3))*1E-4

def objective_text(vertices,objective_vec):
    return ["obj: "+str(sum(vx.flatten()*objective_vec.flatten())) for vx in vertices]
        
def highlight_face(poly,face_num,fig,visible=True):
    colour_face(fig,poly.face_vxs[face_num],face_num=face_num,visible=visible)
    
def compute_range(poly):
    R=poly.get_range()
    spread=R[:,1]-R[:,0]
    offset=0.2*spread
    return np.array([R[:,0]-offset,R[:,1]+offset]).T

def colour_face(fig,face_vxs,face_num=-1,color="lightpink",visible=True,name=None):
    if name is not None:
        name=name
    else:
        name="face_"+str(face_num)
    n=len(face_vxs)
    if n==0:
        return ## no use colouring a face consisting of just an edge or a vertex
    if n==1:
        fig.add_trace(go.Scatter3d(x=face_vxs[:,0],y=face_vxs[:,1],z=face_vxs[:,2],mode="markers",marker=dict(color=color),
                              name=name,visible=visible))
    elif n==2: 
        fig.add_trace(go.Scatter3d(x=face_vxs[:,0],y=face_vxs[:,1],z=face_vxs[:,2],mode="lines",hoverinfo="skip",line=dict(width=40,color=color),
                                  opacity=0.8,name=name,visible=visible))        
    elif n==3: # for some reason alphahull=0 does not work if face is just a triangle
        #print("triangle!")
        fig.add_trace(go.Mesh3d(x=face_vxs[:,0],y=face_vxs[:,1],z=face_vxs[:,2],color=color,hoverinfo="skip",
                            opacity=0.5,flatshading=True,i=[0],j=[1],k=[2],name=name,visible=visible))
    else:
        jitter_vxs=perturb(face_vxs) # Mesh3d has troubles showing flat face
        fig.add_trace(go.Mesh3d(x=jitter_vxs[:,0],y=jitter_vxs[:,1],z=jitter_vxs[:,2],color=color,hoverinfo="skip",
                            opacity=0.5, alphahull=0,flatshading=True,name=name,visible=visible))

def normalise(vec):
    return vec/np.linalg.norm(vec)

def compute_affine_basis(a,beta,scale=1000):
    a=np.array(a).reshape(1,-1)
    beta=np.array(beta).reshape(-1)
    x,_,_,_=np.linalg.lstsq(a,beta,rcond=None)
    a=a.flatten()
    for i in range(3):
        if a[i]!=0:
            break
    vecs=[np.eye(3)[j] for j in range(3) if j!=i]
    a_normalised=normalise(a)
    u=vecs[0]-np.inner(a_normalised,vecs[0])*a_normalised
    u=normalise(u)
    w=vecs[1]-np.inner(a_normalised,vecs[1])*a_normalised-np.inner(u,vecs[1])*u
    w=normalise(w)
    return np.array([x+scale*u,x+scale*w,x-scale*(u+w)])

def setup_ineq_planes(fig,poly,color="lightpink"):
    for i in range(poly.num_ineqs):
        B=compute_affine_basis(poly.A[i],poly.b.flatten()[i])
        fig.add_trace(go.Mesh3d(x=B[:,0],y=B[:,1],z=B[:,2],color=color,hoverinfo="skip",
                            opacity=0.5,flatshading=True,i=[0],j=[1],k=[2],visible=False,name="plane_"+str(i)))
        
def plot_opt_path(fig,opt_path):
    opt_path=np.array(opt_path)
    #fig.add_trace(go.Scatter3d(x=opt_path[:,0],y=opt_path[:,1],z=opt_path[:,2],mode="lines",hoverinfo="skip",line=dict(width=10,color="red"),
    #                          name="opt_path",visible=False))
    fig.add_trace(go.Scatter3d(x=opt_path[0:1,0],y=opt_path[0:1,1],z=opt_path[0:1,2],mode="markers",marker=dict(color="red"),
                              name="start_vx",visible=True))
    for i in range(len(opt_path)-1):
        fig.add_trace(go.Scatter3d(x=opt_path[i:i+2,0],y=opt_path[i:i+2,1],z=opt_path[i:i+2,2],mode="lines",hoverinfo="skip",line=dict(width=10,color="red"),
                                  name="opt_path_"+str(i),visible=False))
        
    
        
def plot_poly(poly,objective_vec=None,opt_path=None,face_traces=False,ineq_planes=False,**kwargs):
    if objective_vec is not None:
        text=objective_text(poly.vertices,objective_vec)
    else:
        text=None
    fig = go.Figure()
    if "range" in kwargs:
        xrange,yrange,zrange=kwargs["range"]
    else:
        xrange,yrange,zrange=compute_range(poly)
    fig.update_layout(
        width=800,
        height=800,
        autosize=False,
        showlegend=False, 
        scene = dict(xaxis = dict(range=xrange), yaxis = dict(range=yrange), zaxis = dict(range=zrange),aspectmode="cube"    ),
        paper_bgcolor="LightSteelBlue",
            margin=dict(
                l=10,
                r=10,
                b=50,
                t=50,
                pad=4
            ),
    )
    fig.add_trace(go.Scatter3d(x=poly.vertices[:,0],y=poly.vertices[:,1],z=poly.vertices[:,2],mode="markers",text=text,name="vertices"))
    for i,edge in enumerate(poly.edges):
        fig.add_trace(go.Scatter3d(x=edge[:,0],y=edge[:,1],z=edge[:,2],marker=dict(size=0), line=dict(width=5,color="blue"),hoverinfo="skip",name="edge_"+str(i)))
    if opt_path is not None:
        plot_opt_path(fig,opt_path)
    colour_face(fig,poly.vertices,color="lightblue",name="whole_poly")
    if face_traces:
        for face_num in range(poly.m):
            highlight_face(poly,face_num,fig,visible=False)
    if ineq_planes:
        setup_ineq_planes(fig,poly)
    #fig.show()
    return fig

def get_path(simplex):
    return [state.x.flatten() for state in simplex.record]

def list_str(l):
    result="["
    for item in l:
        result+=str(round(item,3))+", "
    return result[:-2]+"]"

def with_sign(x):
    r=round(x,2)
    if r<0:
        return str(r)
    return "+"+str(r)

def ineq_string(A,b,ineq_num):
    a=A[ineq_num]
    beta=np.array(b).flatten()[ineq_num]
    return "{}x{}y{}z <= {}".format(round(a[0],2),with_sign(a[1]),with_sign(a[2]),round(beta,2))

def setup_poly_viewer(A,b,*args,**kwargs):
    if "enclose_factor" in kwargs:
        poly=Polyhedron(A,b,enclose_factor=kwargs["enclose_factor"])
    else:
        poly=Polyhedron(A,b)
    fig=plot_poly(poly,face_traces=True,ineq_planes=True,**kwargs)
    none_button=dict(method='restyle',
                     label="none",
                     visible=True,
                     )
    buttons=[none_button]
    ineq_trace_numbers=[]
    for i,trace in enumerate(fig.data):
        if trace.name is not None and trace.name.startswith("plane_"):
            ineq_index=int(trace.name[6:])
            button = dict(method='restyle',
                          label=ineq_string(A,b,ineq_index),
                          visible=True,
                          args=[{'visible':True}, [i]],
                          args2 = [{'visible': False}, [i]],
                         )
            buttons.append(button)
            ineq_trace_numbers.append(i)
    none_button["args"]=[{"visible":False}, ineq_trace_numbers]

    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='down',
                showactive=False,
                buttons=buttons,
                #bgcolor="lightpink",
                bordercolor="darkslategray"
            )
        ],
        margin=dict(
            l=150,
            r=10,
            b=50,
            t=50,
            pad=4
        ),
        title_text="Click to show inequalities"
    )
    fig.show()

#################  code for simplex visualisation #############################

def set_trace_visibility(opt_path_step,trace,smplx,mark_faces=False):
    if trace.name is None:
        return True
    if trace.name=="start_vx":
        return True
    if trace.name.startswith("opt_path_"):
        return int(trace.name[9:]) < opt_path_step
    if mark_faces:
        if trace.name.startswith("face_"):
            J=smplx.record[opt_path_step].J
            return int(trace.name[5:]) in J
        if trace.visible is None:
            return True
    if trace.visible is None:
        return True
    return trace.visible

def complete_y(smplx,opt_path_step):
    m=smplx.m
    y=np.zeros(m)
    J=smplx.record[opt_path_step].J
    y[J]=smplx.record[opt_path_step].y_J
    return y

def state_string(smplx,opt_path_step):
    state=smplx.record[opt_path_step]
    result=""
    result+="Current state of simplex"+"<br>"
    result+="J={}".format(state.J)+"<br>"
    result+="x={}".format(list_str(state.x))+"<br>"
    result+="objective value={}".format(round(state.objective,2))+"<br>"
    result+="y={}".format(list_str(complete_y(smplx,opt_path_step)))+"<br>"
    if state.opt_status!="OPT found":
        result+="i={}  j*={}".format(state.i,state.jstar)  
    else:
        result+="optimum reached"
    return result

def set_visibility(fig,opt_path_step,smplx,mark_faces=False):
    return [set_trace_visibility(opt_path_step,trace,smplx,mark_faces=mark_faces) for trace in fig.data]

def setup_annotations_dict(smplx,opt_path_step):
    annotation = go.layout.Annotation(
        text=state_string(smplx,opt_path_step),
        y=1,
        x=1.2,
        yref="paper",
        xref="paper",
        xanchor="right",
        yanchor="top",
        font=dict(color='black', size=14),
        showarrow=False,
        align="left",
        bgcolor="LightSteelBlue",
        bordercolor="darkslategray",
        borderwidth=2,
    )
    return [annotation]

def setup_simplex_widget(A,b,c,x0):
    poly=Polyhedron(A,b)
    smplx=simplex.Simplex(A,b,c,x0,verbose=True)
    opt_path=get_path(smplx)    
    fig=plot_poly(poly,opt_path=opt_path,face_traces=True,objective_vec=c)

    steps = []
    for i in range(len(opt_path)):
        step = dict(
            method="update",
            args=[{"visible": set_visibility(fig,i,smplx,mark_faces=True)},         
                  {"title": "Simplex step number: " + str(i),
                   "annotations": setup_annotations_dict(smplx,i)}],  # layout attribute
            value=i,
            label=str(i),
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Simplex step: "},
        pad={"t": 50},
        steps=steps
    )]    

    # simulate action of slider step 0
    for trace,visible in zip(fig.data,steps[0]['args'][0]['visible']):
        trace.update(visible=visible)

    fig.update_layout(
        sliders=sliders,
        margin=dict(
        l=10,
        r=150,
        b=50,
        t=50,
        pad=4
        ),
        annotations=setup_annotations_dict(smplx,0),
        title="Simplex step number: 0",
    )

    fig.show() 