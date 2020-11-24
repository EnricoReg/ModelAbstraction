import json
import torch
import numpy as np
#from diffquantitative import DiffQuantitativeSemantic
from scipy.interpolate import griddata, interpolate 
import matplotlib.pyplot as plt
import numpy


class ElMotor:
    def __init__(self):
        
        self.max_speed = 1150
        self.max_torque = 180
        
        self.EM_w_list=np.array([0,95,190,285,380,475,570,665,760,855,950,1045,1140])
        self.EM_T_list=np.array([0,11.25,22.5,33.75,45,56.25,67.5,78.75,90,101.25,112.5,123.75,135,146.25,157.5,168.75,180])
       
        x2d, y2d = np.meshgrid(self.EM_w_list, self.EM_T_list)
       
        self.x2d = x2d
        self.y2d = y2d
       
        self.x_speed_flat = x2d.flatten()
        self.y_torque_flat = y2d.flatten()
       
        #self.EM_T_max_list   = np.array([179.1,179,180.05,180,174.76,174.76,165.13,147.78,147.78,109.68,109.68,84.46,84.46])
        self.EM_T_max_list   = np.array([180,180,180,180,174.76,170,165.13,150,137.78,115.68,105.68,94.46,84.46])
        
        self.f_max_rq = interpolate.interp1d(self.EM_w_list, self.EM_T_max_list, kind =  "cubic", fill_value="extrapolate")

        self.efficiency = np.array([
        [.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50],
        [.68,.70,.71,.71,.71,.71,.70,.70,.69,.69,.69,.68,.67,.67,.67,.67,.67],
        [.68,.75,.80,.81,.81,.81,.81,.81,.81,.81,.80,.80,.79,.78,.77,.76,.76],
        [.68,.77,.81,.85,.85,.85,.85,.85,.85,.84,.84,.83,.83,.82,.82,.80,.79],
        [.68,.78,.82,.87,.88,.88,.88,.88,.88,.87,.87,.86,.86,.85,.84,.83,.83],
        [.68,.78,.82,.88,.88,.89,.89,.89,.88,.88,.87,.85,.85,.84,.84,.84,.83],
        [.69,.78,.83,.87,.88,.89,.89,.88,.87,.85,.85,.84,.84,.84,.84,.84,.83],
        [.69,.73,.82,.86,.87,.88,.87,.86,.85,.84,.84,.84,.84,.84,.84,.84,.83],
        [.69,.71,.80,.83,.85,.86,.85,.85,.84,.84,.84,.84,.84,.84,.83,.83,.83],
        [.69,.69,.79,.82,.84,.84,.84,.84,.83,.83,.83,.83,.83,.83,.83,.82,.82],
        [.69,.68,.75,.81,.82,.81,.81,.81,.81,.81,.81,.80,.80,.80,.80,.80,.80],
        [.69,.68,.73,.80,.81,.80,.76,.76,.76,.76,.76,.76,.76,.76,.75,.75,.75],
        [.69,.68,.71,.75,.75,.75,.75,.75,.75,.75,.75,.75,.74,.74,.74,.74,.74] ]).T
       
        
       
        self.efficiency_flat = self.efficiency.flatten()
        self.get_eff_matrix()
       
    def getEfficiency(self, speed, torque): 
        points = (self.x_speed_flat, self.y_torque_flat)
        pair = (speed, torque)
        grid = griddata(points, self.efficiency_flat, pair, method = "cubic")
        # todo: debug
        grid[torque > self.f_max_rq(speed) ] = np.nan
        
        # print(grid)
        return grid
   
    def getMinMaxTorque(self, speed):
        max_tq = numpy.interp(speed.cpu().detach().numpy(), self.EM_w_list, self.EM_T_max_list)
        return -max_tq[0], max_tq[0]

    def get_eff_matrix(self):
        
        self.speed_vect = np.linspace(0,self.max_speed,201)
        self.torque_vect = np.linspace(0,self.max_torque,151)
        xx, yy = np.meshgrid(self.speed_vect, self.torque_vect)

        self.eff_matrix = self.getEfficiency(xx, yy) #.reshape((emot.speed_vect.shape[0],emot.torque_vect.shape[0]))
                
        self.eff_matrix[yy >  self.f_max_rq(xx) ] = np.nan

    def plotEffMap(self):

        """
        fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        ax = plt.contourf(self.x2d, self.y2d, self.efficiency, cmap = 'jet')
        #ax1.plot(self.EM_w_list,self.EM_T_max_list )
        plt.colorbar(ax)
        plt.show()
        """
        
        fig1 = plt.figure()
        ax1 = plt.contourf(self.speed_vect, self.torque_vect, self.eff_matrix,levels = 30 ,cmap = 'jet')
        
        plt.plot(self.EM_w_list,self.EM_T_max_list , 'k')
        plt.colorbar(ax1)
        plt.show()

        
#%%

emot = ElMotor()
emot.plotEffMap()   

#%%
class Car:
    """ Describes the physical behaviour of the vehicle """
    def __init__(self, device):
        self.device=device

        self._max_acceleration = 3.0   #m/s^2
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 20.0 #m/s
        self._min_velocity = 0.0
        self.gravity = 9.81 #m/s^2
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.01 # will be ignored

        self.mass = 800       #kg
        self.rho =  1.22          #the air density, 
        self.aer_coeff = 0.4     #the aerodynamic coefficient
        self.veh_surface  = 2  #equivalent vehicle surface
        self.rr_coeff =  8*10^-3     #rolling resistance coefficient
        self.gear_ratio = 10
        self.wheel_radius = 0.3  #effective wheel radius
        self._max_whl_brk_torque = 2000  #Nm
        
        self.e_motor = ElMotor()
        
        self.max_e_tq = np.max(self.e_motor.EM_T_max_list)
        self.min_e_tq = - self.max_e_tq
        self.e_motor_speed = torch.tensor(0.0)
        self.e_torque= torch.tensor(0.0)
        self.br_torque= torch.tensor(0.0)
        self.e_power = torch.tensor(0.0)

    def motor_efficiency(self):
        eff = self.e_motor.getEfficiency(self.e_motor_speed.item(),self.e_torque.item())
        return eff**(-torch.sign(self.e_torque))
    
    def calculate_wheels_torque(self, e_torque, br_torque):
        self.br_torque = torch.clamp(br_torque, 0, self._max_whl_brk_torque)
        self.e_torque = torch.clamp(e_torque, self.min_e_tq, self.max_e_tq)
        return self.e_torque*self.gear_ratio - self.br_torque

    def resistance_force(self):
        F_loss = 0.5*self.rho*self.veh_surface*self.aer_coeff*(self.velocity**2) + \
            self.rr_coeff*self.mass*self.gravity*self.velocity
        return F_loss


    def update(self, dt, e_torque, br_torque, dist_force=0):
        #Differential equation for updating the state of the car

        in_wheels_torque = self.calculate_wheels_torque(e_torque, br_torque)

        acceleration = (in_wheels_torque/self.wheel_radius - self.resistance_force() + dist_force) / self.mass
           
        self.acceleration = torch.clamp(acceleration, self._min_acceleration, self._max_acceleration)
        
        # self.velocity = torch.clamp(self.velocity + self.acceleration * dt, self._min_velocity, self._max_velocity)
        self.velocity = self.velocity + self.acceleration * dt
        self.e_motor_speed = self.velocity*self.gear_ratio/self.wheel_radius
        
        #update min/max e-torque based on new motor speed
        self.min_e_tq, self.max_e_tq = self.e_motor.getMinMaxTorque(self.e_motor_speed)
        # update power consumed
        self.e_power = self.e_motor_speed*self.e_torque*self.motor_efficiency().item()        
        self.position += self.velocity * dt

        print(f"pos={self.position.item()}\tpower={self.e_power.item()}")

