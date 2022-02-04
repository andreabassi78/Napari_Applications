# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:28:49 2022

@author: andrea
"""
import napari
from qtpy.QtWidgets import QLabel, QVBoxLayout,QSplitter, QHBoxLayout, QWidget, QPushButton
from napari.layers import Image
import numpy as np
from numpy.fft import ifft2, fft2, fftshift 
from napari.qt.threading import thread_worker
from magicgui import magicgui
import warnings
import pathlib
import random

pi = np.pi

       
# class _Vector():
#     def __init__(self,initial_pos=[0.,0.,0.]):
#         self._pos = np.array(initial_pos)
       
#     @property    
#     def pos(self):
#         return self._pos
    
#     @pos.setter 
#     def pos(self, new_pos):
#         self._pos = new_pos
    
#     @property    
#     def x(self):
#         print(self._pos[2])
#         return self._pos[2]
    
#     @pos.setter 
#     def x(self, new_x):
#         self._pos[2] = new_x
    
#     @property    
#     def y(self):
#         return self._pos[1]
    
#     @pos.setter 
#     def y(self, new_y):
#         self._pos[1] = new_y
    
#     @property    
#     def z(self):
#         return self._pos[0]
    
#     @pos.setter 
#     def z(self, new_z):
#         self._pos[0] = new_z     
        
class Sphere():
    def __init__(self, viewer):
        self._pos = np.zeros(3)
        self.pointlayer = viewer.layers['mypoints'].add(self._pos*100)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        
    @property    
    def position(self):
        return self._pos 
    
    @position.setter 
    def position(self, new_pos):
        #self.pointlayer.data = new_pos*100
        self._pos = new_pos
    
    @property    
    def color(self):
        return self._color 
    
    @color.setter 
    def color(self, new_color):
        self._color = new_color
    


class System:
           
    def __init__(self, viewer, ms, r0s, v0s, dt):
        self.bodies = []
        self.collided_couples = []
        self.attached_couples = []
        self.dt = dt
        # Create the masses  
        for index, m in enumerate(ms):
            body =  Sphere(viewer)
            body.mass = m
            body.radius = self.set_radius(m)
            body.position = r0s[index]
            body.velocity = v0s[index]
            body.acceleration = np.array([0,0,0])
            body.color =  np.array([1,random.random(),random.random()])
            self.bodies.append(body)
    
    
    def set_radius(self, mass, density = 200):
        # Draw a sphere with a radius proportional to the cubic root of the volume
        # density is in kg/m^3
        radius = (3/(4*pi)*mass/density)**(1/3) 
        return radius
    
    
    def set_position(self):
        for index, b in enumerate(self.bodies):
            b.position = b.position + b.velocity*self.dt
    
    
    def set_velocity(self):
        for index, b in enumerate(self.bodies):
            b.velocity = b.velocity + b.acceleration.pos*self.dt
              
            
    def check_collisions(self):
        for index,body in enumerate(self.bodies):            
            for other_index, other_body in enumerate(self.bodies) : 
                if index != other_index:
                    distance =  mag(body.position-other_body.position)
                    if distance <= (body.radius+other_body.radius):
                        if [other_index, index] not in self.collided_couples: 
                            self.collided_couples.append([index,other_index])
        return(self.collided_couples)
                      
                                              
    def elastic_collision(self):
        for indexs in self.collided_couples: 
            if indexs in self.attached_couples: continue
            body0 = self.bodies[indexs[0]]
            body1 = self.bodies[indexs[1]]
            vrel = body0.velocity - body1.velocity
            rrel = body0.position-body1.position
            a = rrel.mag2 # magnitude squared
            ratio0 = 2 * body1.mass / (body0.mass + body1.mass) 
            ratio1 = 2 * body0.mass / (body0.mass + body1.mass) 
            body0.velocity += - ratio0 * np.dot(vrel,rrel) / a  *rrel 
            body1.velocity += - ratio1 * np.dot(-vrel,-rrel) / a  *(-rrel)
        self.attached_couples = self.collided_couples.copy()    
        self.collided_couples.clear()
        
        
    def totally_inelastic_collision(self):
        bodies_to_remove = []
        for indexs in self.collided_couples: 
            body0 = self.bodies[indexs[0]]
            body1 = self.bodies[indexs[1]]
            m0 = body0.mass
            m1 = body1.mass
            if m0 ==0 and m1 == 0: continue
            # grow body 1 
            body0.position = (body0.position*m0+body1.position*m1)/(m0+m1)
            body0.velocity = (body0.velocity*m0+body1.velocity*m1)/(m0+m1)
            body0.mass += body1.mass
            body0.radius = self.set_radius(body0.mass)
            # body2: set the mass to 0 
            body1.velocity = body0.velocity
            body1.mass = 0
            body1.visible = False
            body1.radius = self.set_radius(body1.mass)
            bodies_to_remove.append(body1)
        for body in bodies_to_remove:
            if body in self.bodies:
                self.bodies.remove(body)
        self.collided_couples.clear()
  
        
    def inelastic_collision(self, K=40, B=1):
        # balls are modeled as elastic (springs) with a shear friction
        # K elastic constant
        # B damping coefficient # if B=0 collision is elastic
        acceleration0 = np.array([0,0,0])
        acceleration1 = np.array([0,0,0])
        for indexs in self.collided_couples: 
            body0 = self.bodies[indexs[0]]
            body1 = self.bodies[indexs[1]]
            vrel = body0.velocity - body1.velocity
            rrel = body0.position-body1.position
            distance =  np.abs(rrel)
            F = + K * rrel / distance * (body0.radius+body1.radius-distance)  - B * vrel
            acceleration0 = + F / body0.mass
            acceleration1 = - F / body1.mass
            body0.velocity += acceleration0*self.dt
            body1.velocity += acceleration1*self.dt    
        self.collided_couples.clear()
        
        
    def partially_inelastic_collision(self):
        # totally inelastic collision, but if the couple of bodies collides with others can breack 
        for indexs in self.collided_couples: 
            body0 = self.bodies[indexs[0]]
            body1 = self.bodies[indexs[1]]
            m0 = body0.mass
            m1 = body1.mass
            # grow body 1 
            cm_pos = (body0.position*m0+body1.position*m1)/(m0+m1)
            cm_velocity = (body0.velocity*m0+body1.velocity*m1)/(m0+m1)            
            dr0 = body0.position-cm_pos
            dv0 = body0.velocity-cm_velocity 
            dr1 = body1.position-cm_pos
            dv1 = body1.velocity-cm_velocity            
            omega0 = np.cross(-dv0,dr0)/np.abs(dr0)**2            
            omega1 = np.cross(-dv1,dr1)/np.abs(dr1)**2
            #assert omega0 == omega1
            body0.velocity = cm_velocity + np.cross(omega0, dr0)
            body1.velocity = cm_velocity + np.cross(omega1, dr1)
        self.collided_couples.clear()
        
        
    def bounce_on_border(self, L = 2):     
        for index,body in enumerate(self.bodies):
            loc = body.position
            
            vel = body.velocity
            if abs(loc[2]) > L/2:
                if loc[2] < 0: vel.x =  abs(vel[2])
                else: vel[2] =  -abs(vel[2])
            if abs(loc[1]) > L/2:
                if loc[1] < 0: vel[1] = abs(vel[1])
                else: vel[1] =  -abs(vel[1])
            if abs(loc[0]) > L/2:
                if loc[0] < 0: vel[0] =  abs(vel[0])
                else: vel[0] =  -abs(vel[0])
                
    
      
         
if __name__ == '__main__':
       
    import time
    L =300
    viewer = napari.Viewer()
    viewer.add_points([],ndim = 3, name='mypoints',size =1, edge_color = 'red')
    viewer.layers['mypoints'].add(np.random.random([100,3])*np.array([15,434,400]))
    current_step = [L/2]*3
    viewer.dims.current_step = current_step
    napari.run()
    
    # L=1
    # # Set temporal sampling in seconds 
    # delta_t = 0.005 
    
    # N =30 # number of bodies
    
        
    # m0 = 25 # kg. The first mass 
    # pos0 = np.array([0.1,0.,0.]) # it starts from the center 
    # v0 = np.array([0.2,0.5,0.]) # it is at rest
    
    # masses =[m0] # list of masses, initialized with the first element 
    # initial_positions = [pos0]  
    # initial_velocities = [v0]
    
    # for idx in range(1,10): 
    #     # Each body has the same mass
    #     masses.append(1)   # kg  
    #     # Place the masses randomly in space
    #     initial_positions.append(np.random.random(3)) # m
    #     # Give random initial velocities
    #     initial_velocities.append(np.random.random(3)) # m/s
        
    
    # sys = System(viewer, masses, initial_positions, initial_velocities, delta_t)
    
    # t = 0
        
    # while t<20:   
    #     t+=1
    #     time.sleep(0.01)
        
    #     sys.set_position()
    #     sys.bounce_on_border(L)        
        
    #     sys.check_collisions()
    #     sys.totally_inelastic_collision()
    # print(sys.bodies[0].position)
        