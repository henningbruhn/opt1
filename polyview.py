
import cdd # pycddlib
import numpy as np
import plotly.graph_objects as go
import math
import ipywidgets as widgets
import simplex

###################### compute vertices and edges of polyhedron ##################

def H_rep(A,b):
    return np.hstack([b,-A])

def find_face_vxs(vxs,poly,face_num,eps=1E-8):
    ineq=np.array(poly.get_inequalities())[face_num]
    face_vxs=[]
    for vx in vxs:
        if np.abs(ineq[0]+np.sum(ineq[1:]*vx))<eps:
            face_vxs.append(vx)
    return np.array(face_vxs)

class Polyhedron:
    def __init__(self,A,b,enclose_factor=1000):
        A=np.array(A)
        b=np.array(b).reshape(-1,1)
        self.A=A
        self.b=b
        if enclose_factor>0:
            upper_bounds_A=np.eye(3)
            upper_bounds_b=enclose_factor*(np.ones(3).reshape(-1,1))
            lower_bounds_A=-np.eye(3)
            lower_bounds_b=enclose_factor*(np.ones(3).reshape(-1,1))
            A=np.vstack([A,upper_bounds_A,lower_bounds_A])
            b=np.vstack([b,upper_bounds_b,lower_bounds_b])
        self.cdd_poly=self._get_polyhedron(A,b)
        self.vertices=self._get_vertices()
        self.edges=self._get_edges()
        self.m,self.n=self.A.shape
        self.face_vxs=[find_face_vxs(self.vertices,self.cdd_poly,face_num) for face_num in range(self.m)]
        
    def get_range(self):
        return np.array([np.min(self.vertices,axis=0),np.max(self.vertices,axis=0)]).T
        
    def _get_polyhedron(self,A,b):
        matrix=cdd.Matrix(H_rep(A,b))
        matrix.rep_type=cdd.RepType.INEQUALITY
        return cdd.Polyhedron(matrix)
    
    def _get_vertices(self):
        generators=np.array(self.cdd_poly.get_generators())
        if len(generators)==0:
            raise Exception("polyhedron empty!")
        vertices=generators[generators[:,0]==1]
        vertices=vertices[:,1:]
        if len(vertices)!=len(generators):
            raise Exception("polyhedron unbounded!")
        return vertices
        
    def _get_edges(self):
        edges=[]
        for vx_index,neighbours in enumerate(self.cdd_poly.get_adjacency()):
            for neighbour_index in neighbours:
                if vx_index<neighbour_index: # don't add edges twice
                    edges.append(np.array([self.vertices[vx_index],self.vertices[neighbour_index]]))
        return edges
    
        

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
    for i in range(poly.m):
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
        text=objective_text(vxs,objective_vec)
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
    fig.add_trace(go.Scatter3d(x=poly.vertices[:,0],y=poly.vertices[:,1],z=poly.vertices[:,2],mode="markers",text=text))
    for edge in poly.edges:
        fig.add_trace(go.Scatter3d(x=edge[:,0],y=edge[:,1],z=edge[:,2],marker=dict(size=0), line=dict(width=5,color="blue"),hoverinfo="skip"))
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

def factory_plot(factory,plot_path=False,face_num=None):
    A,b,c,x0=factory()
    poly=get_polyhedron(A,b)
    if plot_path:
        simplex=Simplex(A,b,c,x0,verbose=True)
        plot_poly(poly,objective_vec=c,opt_path=get_path(simplex),face_num=face_num)
    else:
        plot_poly(poly,face_num=face_num)-1
        
        
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
    out=widgets.Output(layout={'border':'6px solid LightSteelBlue','width': '90%', 
                'height': '60px',})
    fig=plot_poly(poly,face_traces=True,ineq_planes=True,**kwargs)
    fig_widget=go.FigureWidget(fig)
    m,n=poly.m,poly.n
    buttons=[
        widgets.Checkbox(
            value=False,
            description=ineq_string(A,b,i),
            disabled=False,
            indent=False
        ) for i in range(m)
    ]
    def face_output(face_num):
        num_vxs_in_face=len(poly.face_vxs[face_num])
        if num_vxs_in_face==0:
            return "empty intersection"
        if num_vxs_in_face==1:
            return "vertex intersection"
        if num_vxs_in_face==2:
            return "edge intersection"
        if num_vxs_in_face>=3:
            return "facet intersection"

    def on_check_factory(ineq_num):
        def on_check(event):
            with fig_widget.batch_update():
                for trace in fig_widget.data:
                    if show_planes:
                        if trace.name[:5]=="plane":
                            if int(trace.name[6:])==ineq_num:
                                trace.update(visible=event.new)
                    else:
                        if trace.name[:4]=="face":
                            if int(trace.name[5:])==ineq_num:
                                trace.update(visible=event.new)
            out.clear_output()
            with out:
                if event.new:
                    print("inequality {} selected".format(ineq_num))
                    print(face_output(ineq_num))
                else:
                    print("no inequality selected")
        return on_check

    for i,button in enumerate(buttons):
        button.observe(on_check_factory(i),"value")
        
    toggle_button=widgets.ToggleButton(
        value=False,
        description='full plane',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='toggles between full plan or intersection with polytope',
        #icon='check' # (FontAwesome names without the `fa-` prefix)
    )
    show_planes=False

    def reset_all():
        with fig_widget.batch_update():
            for trace in fig_widget.data:
                if trace.name[:5]=="plane" or trace.name[:4]=="face":
                    trace.update(visible=False)
        for btn in buttons:
            btn.value=False
    
    def toggle_action(event):
        nonlocal show_planes
        show_planes=event.new
        reset_all()
    
    toggle_button.observe(toggle_action,"value")

    explainer=widgets.HTML(
        value="Select inequalities to highlight intersection",
        placeholder='Some HTML',
        #description='Some HTML',
    )    
    vbox=widgets.VBox([explainer]+[toggle_button]+buttons+[out])
    hbox=widgets.HBox([fig_widget,vbox])
    with out:
        print("no inequality selected")

    return hbox    


class Display_State:
    def __init__(self,smplx,fig_widget,output,face_highlighting=True):
        self.simplex=smplx
        self.step_num=-1
        self.max_step_num=len(self.simplex.record)
        self.widget=fig_widget
        self.out=output
        self.forward_button=widgets.Button(icon="forward")
        self.backward_button=widgets.Button(icon="backward",disabled=True)
        self.forward_button.on_click(self.forward_func())
        self.backward_button.on_click(self.backward_func())
        self.do_face_highlighting=face_highlighting
        
    def complete_y(self,y_J,J):
        m=self.simplex.m
        y=np.zeros(m)
        y[J]=y_J
        return y
    
    def print_state(self):
        self.out.clear_output()
        with self.out:
            print("Step number: {}".format(self.step_num))
            state=self.simplex.record[self.step_num]
            print("J={}".format(state.J))
            print("x={}".format(list_str(state.x)))
            print("objective value={}".format(round(state.objective,2)))
            y=self.complete_y(state.y_J,state.J)
            print("y={}".format(list_str(y)))
            if state.opt_status!="OPT found":
                print("i | j*= {} | {}".format(state.i,state.jstar))  
            else:
                print("optimum reached")

    def update_path(self):
        with self.widget.batch_update():
            for trace in self.widget.data:
                if trace.name[:8]=="opt_path":
                    if int(trace.name[9:])<self.step_num:
                        trace.update(visible=True)
                    else:
                        trace.update(visible=False)
            if self.do_face_highlighting:
                self.update_face_highlighting()
                        
    def update_face_highlighting(self):
        state=self.simplex.record[self.step_num]
        J=state.J
        with self.widget.batch_update():
            for trace in self.widget.data:
                if trace.name[:4]=="face":
                    if int(trace.name[5:]) in J:
                        trace.update(visible=True)
                    else:
                        trace.update(visible=False)

    def update_button_disable(self):
        if self.step_num==self.max_step_num-1:
            self.forward_button.disabled=True
        if self.step_num>0:
            self.backward_button.disabled=False
        
                        
    def forward_func(self):
        def forward(button):
            if self.step_num<self.max_step_num-1:
                self.step_num+=1
                self.print_state()
                self.update_path()
            self.update_button_disable()
        return forward
            
    def backward_func(self):
        def backward(button):
            if self.step_num>0:
                self.forward_button.disabled=False
                self.step_num-=1
                self.print_state()
                self.update_path()
            self.update_button_disable()
        return backward
    
def setup_simplex_widget(A,b,c,x0):
    poly=Polyhedron(A,b)

    smplx=simplex.Simplex(A,b,c,x0,verbose=True)
    opt_path=get_path(smplx)    
    out=widgets.Output(layout={'border':'6px solid LightSteelBlue','width': '90%', 
                'height': '160px',})

    fig=plot_poly(poly,opt_path=opt_path,face_traces=True)
    fig_widget=go.FigureWidget(fig)

    dstate=Display_State(smplx,fig_widget,out)
    forward=dstate.forward_button
    backward=dstate.backward_button
    button_box=widgets.HBox([backward,forward])
    vbox=widgets.VBox([button_box,out])
    hbox=widgets.HBox([fig_widget,vbox])
    with out:
        print("Click buttons to perform simplex step")

    return hbox