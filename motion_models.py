import numpy as np

def transform_trajectory(rel_poses,start_pose):
    cs,ss = np.cos(start_pose[2]), np.sin(start_pose[2])
    rmat = np.array([[cs,ss,0],[-ss, cs,0],[0,0,1]],dtype=float)
    return np.dot(rel_poses, rmat)+start_pose
        
class yaw_rate_motion:
    def __init__(self, max_yaw=np.pi, speed=1.0, n_yaws=5, n_points=11):
        self.max_yaw = max_yaw
        self.speed = speed
        self.n_yaws = n_yaws
        self.n_points = n_points
        self.full_pose = np.zeros((n_points,3,n_yaws))
        self.yaw_rates = np.linspace(-max_yaw, max_yaw, n_yaws)
        self.t = np.linspace(0, 1.0, n_points)
        
        for ii,yd in enumerate(self.yaw_rates):
            if yd != 0.0:
                theta = self.t*yd - np.pi
                r = 1.0/yd
                y = r + r*np.cos(theta)
                x = -r*np.sin(theta)
            else:
                y = np.zeros(n_points)
                x = np.linspace(0,1.0,n_points)
            self.full_pose[:,0,ii] = x*speed
            self.full_pose[:,1,ii] = y*speed
            self.full_pose[:,2,ii] = np.linspace(0,yd,n_points)
            
    def get_paths_number(self):
        return self.n_yaws
        
    def get_end_points(self,start_pose=np.array([0.0,0.0,0.0])):
        rel_poses = self.full_pose[-1,:,:].transpose()
        return transform_trajectory(rel_poses,start_pose)
        
    def get_trajectory(self,start_pose=np.array([0.0,0.0,0.0]),ii=0):
        rel_poses = self.full_pose[:,:,ii]
        return transform_trajectory(rel_poses,start_pose)