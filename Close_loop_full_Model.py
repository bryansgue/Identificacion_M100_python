import sys
import time
import rospy
import os
import numpy as np
from scipy.signal import savgol_filter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R
from scipy.io import savemat

## Global variables system
xd = 3.0
yd = -4.6
zd = 5.16
vxd = 0.0
vyd = 0.0
vzd = 0.0

qx = 0.0005
qy = 0.0
qz = 0.0
qw = 1.0
wxd = 0.0
wyd = 0.0
wzd = 0.0



def get_quaternios():
    quat = np.array([qw, qx, qy, qz], dtype=np.double)
    return quat

def get_euler():
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    eul = rot.as_euler('xyz', degrees=False)

    euler = np.array([eul[0], eul[1], eul[2]], dtype=np.double)
    return euler

def get_euler_p(omega, euler):
    W = np.array([[1, np.sin(euler[0])*np.tan(euler[1]), np.cos(euler[0])*np.tan(euler[1])],
                  [0, np.cos(euler[0]), np.sin(euler[0])],
                  [0, np.sin(euler[0])/np.cos(euler[1]), np.cos(euler[0])/np.cos(euler[1])]])

    euler_p = np.dot(W, omega)
    return euler_p

def get_pos():
    h = np.array([xd, yd, zd], dtype=np.double)
    return h

def get_inertial_vel():
    v = np.array([vxd, vyd, vzd], dtype=np.double)
    return v

def get_body_vel():
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    rot = rot.as_matrix()
    rot_inv = np.linalg.inv(rot)
    vel_w = np.array([vxd, vyd, vzd], dtype=np.double)
    vel_w = vel_w.reshape(3,1)
    vel = rot_inv@vel_w 

    u = np.array([vel[0,0], vel[1,0], vel[2,0]], dtype=np.double)
    return u

def get_omega():
    omega = np.array([wxd, wyd, wzd], dtype=np.double)
    return omega
# Get system velocities


## Reference system
def send_reference(ref,ref_pub, ref_msg):
        ref_msg.twist.linear.x = 0
        ref_msg.twist.linear.y = 0
        ref_msg.twist.linear.z = ref[0]

        ref_msg.twist.angular.x = ref[1]
        ref_msg.twist.angular.y = ref[2]
        ref_msg.twist.angular.z = ref[3]


        # Publish control values
        ref_pub.publish(ref_msg)
        

def odometry_call_back(odom_msg):
    global xd, yd, zd, qx, qy, qz, qw, vxd, vyd, vzd, wxd, wyd, wzd, time_message

    # Read desired linear velocities from node
    time_message = odom_msg.header.stamp
    xd = odom_msg.pose.pose.position.x 
    yd = odom_msg.pose.pose.position.y
    zd = odom_msg.pose.pose.position.z
    vxd = odom_msg.twist.twist.linear.x
    vyd = odom_msg.twist.twist.linear.y
    vzd = odom_msg.twist.twist.linear.z


    qx = odom_msg.pose.pose.orientation.x
    qy = odom_msg.pose.pose.orientation.y
    qz = odom_msg.pose.pose.orientation.z
    qw = odom_msg.pose.pose.orientation.w

    wxd = odom_msg.twist.twist.angular.x
    wyd = odom_msg.twist.twist.angular.y
    wzd = odom_msg.twist.twist.angular.z
    return None

def PID_function(vz_d, vz, Kp, Kd, Ki, error_history, error_prev):
    # Calcular el error en el instante actual
    error = np.tanh(vz_d - vz)
    
    # Calcular la acción proporcional
    u_P = Kp * error
    
    # Calcular la acción derivativa
    
    u_D = Kd * (error - error_prev)
    
    # Calcular la acción integral
    accumulated_error = error_history
    u_I = Ki * accumulated_error
    
    # Calcular la señal de control total
    u_total = u_P + u_D + u_I 
    
    return u_total

def main(control_pub, ref_msg):

    # Twist 
    ref_drone = TwistStamped()

    # Simulation time parameters
    ts = 1/30
    tf = 30
    t = np.arange(0, tf+ts, ts, dtype=np.double)

    # States System pose
    states = np.zeros((22, t.shape[0]+1), dtype=np.double)

    # BOdy velocity
    h = np.zeros((3, t.shape[0]+1), dtype=np.double)
    euler = np.zeros((3, t.shape[0]+1), dtype=np.double) 
    v = np.zeros((3, t.shape[0]+1), dtype=np.double)
    euler_p = np.zeros((3, t.shape[0]+1), dtype=np.double) 
    omega = np.zeros((3, t.shape[0]+1), dtype=np.double)
    quat = np.zeros((4, t.shape[0]+1), dtype=np.double)
    u = np.zeros((3, t.shape[0]+1), dtype=np.double)

    # Controlador Cinematico
    hd = np.zeros((4, t.shape[0]+1), dtype=np.double)
    hdp = np.zeros((4, t.shape[0]+1), dtype=np.double)
    he = np.zeros((4, t.shape[0]+1), dtype=np.double)
    uc = np.zeros((4, t.shape[0]+1), dtype=np.double)
    psidp = np.zeros((1, t.shape[0]+1), dtype=np.double)

    J = np.zeros((4, 4))
    K1 = np.diag([1,1,1,1])  # Distribuir los primeros 4 elementos de Gains en la matriz K1
    K2 = np.diag([1,1,1,1]) 

    
    # Control signals
    T_ref = np.zeros((4, t.shape[0]), dtype=np.double)

    # Define Control Action

    print("Sin experimeto")
    T_ref[0, :]=  0
    T_ref[1, :] = 0
    T_ref[2, :] = 0
    T_ref[3, :] = 0
    

    # INICIALIZA EL EXPERIMENTO
    for k in range(0, 50):
        tic = time.time()
        
        while (time.time() - tic <= ts):
                None
        
        # Save Data
        h[:, 0] = get_pos()
        euler[:, 0] = get_euler()
        v[:, 0] = get_inertial_vel()      
        omega[:, 0] = get_omega()
        euler_p[:, 0] = get_euler_p(omega[:, 0],euler[:, 0])
        quat[:, 0] = get_quaternios()
        u[:, 0] = get_body_vel()

        print("Initializing the experiment")
    

    # PID parameters
    error_history_ux = 0
    error_prev_ux = 0

    error_history_uy = 0
    error_prev_uy = 0

    # Parámetros del controlador PDI
    Kp_ux = 0.25
    Kd_ux = 0.001
    Ki_ux = 0.006

    Kp_uy = 0.25
    Kd_uy = 0.001
    Ki_uy = 0.006

    # Tarea deseada
    experiment_number = 12

    if experiment_number == 11:
        xref = lambda t: 5 * np.sin(8*0.04 * t) + 0.1
        yref = lambda t: 5 * np.sin(8*0.08 * t) + 0.1
        zref = lambda t: 3 * np.sin(0.25 * t) + 10

        xref_p = lambda t: 5 *7* 0.04 * np.cos(8*0.04 * t)
        yref_p = lambda t: 5 *7* 0.08 * np.cos(8*0.08 * t)
        zref_p = lambda t: 3*0.1 * np.cos(0.25 * t)

        xref_pp = lambda t: -5 *8* 0.04 * 0.04 * np.sin(8*0.04 * t)
        yref_pp = lambda t: -5 *8* 0.08 * 0.08 * np.sin(8*0.08 * t)

    elif experiment_number == 12:
        xref = lambda t: (5 * np.sin(8 * 0.04 * t) + 0.1) * np.cos(0.2 * t)
        yref = lambda t: (5 * np.sin(8 * 0.08 * t) + 0.1) * np.sin(0.2 * t)
        zref = lambda t: 3 * np.sin(0.25 * t) + 10

        xref_p = lambda t: 5 *7* 0.04 * np.cos(8*0.04 * t)
        yref_p = lambda t: 5 *7* 0.08 * np.cos(8*0.08 * t)
        zref_p = lambda t: 3*0.1 * np.cos(0.25 * t)

        xref_pp = lambda t: -5 *8* 0.04 * 0.04 * np.sin(8*0.04 * t)
        yref_pp = lambda t: -5 *8* 0.08 * 0.08 * np.sin(8*0.08 * t)
    else:
        print("Sin experimeto")

    hxd = xref(t)
    hyd = yref(t)
    hzd = zref(t)

    hxdp = xref_p(t)
    hydp = yref_p(t)
    hzdp = zref_p(t)

    hxdpp = xref_pp(t)
    hydpp = yref_pp(t)

    psid = np.arctan2(hydp, hxdp)
    
    max_psidp = (5/6) * np.pi

    for k in range(0, t.shape[0]):
        if k > 0:
            delta_psidp = (psid[k] - psid[k - 1]) / ts
            # Aplicar la saturación a delta_psidp
            delta_psidp = min(max_psidp, max(-max_psidp, delta_psidp))
            psidp[0, k] = delta_psidp
        else:
            psidp[0, k] = psid[k] / ts

    psidp[0, 0] = 0

    
    # COMEMIENZA LA IDENTIFICACION
    for k in range(0, t.shape[0]):
        tic = time.time()

        # Controlador de bajo nivel 
        hd[:, k] = [hxd[k], hyd[k], hzd[k], psid[k]]
        hdp[:, k] = [hxdp[k], hydp[k], hzdp[k], psidp[0,k]]


        psi = euler[2, k]
        J[0, 0] = np.cos(psi)
        J[0, 1] = -np.sin(psi)
        J[1, 0] = np.sin(psi)
        J[1, 1] = np.cos(psi)
        J[2, 2] = 1
        J[3, 3] = 1

        he[:, k] = hd[:, k] -  np.hstack((h[:, k], psi))
        uc[:, k] = np.linalg.pinv(J) @ (K1 @ np.tanh(K2 @ he[:, k]))
        
        # PID UX
        ux = u[0,k]
        ux_ref = uc[0, k]
        error_ux = np.tanh(ux_ref - ux)
        error_history_ux = error_history_ux + error_ux
        theta_PID  = PID_function(ux_ref, ux, Kp_ux, Kd_ux, Ki_ux, error_history_ux, error_prev_ux)

        # PID VY
        uy = u[1,k]
        uy_ref = uc[1, k]
        error_uy = np.tanh(uy_ref - uy)
        error_history_uy = error_history_uy + error_uy
        phi_PID  = PID_function(uy_ref, uy, Kp_uy, Kd_uy, Ki_uy, error_history_uy, error_prev_uy)
        

        # Send Control action to the system
        T_ref[:, k] = [uc[2, k] , -phi_PID, theta_PID, uc[3, k]]
        print(ux,uy)
        send_reference(T_ref[:, k], control_pub, ref_msg)


        # Loop_rate.sleep()
        while (time.time() - tic <= ts):
                None
        toc = time.time() - tic 

        #ERROR PID
        error_prev_ux = error_ux
        error_prev_uy = error_uy

        #print(toc)

        # Save Data
        h[:, k+1] = get_pos()
        euler[:, k+1] = get_euler()
        v[:, k+1] = get_inertial_vel()     
        omega[:, k+1] = get_omega()
        euler_p[:, k+1] = get_euler_p(omega[:, k+1],euler[:, k+1])
        quat[:, k+1] = get_quaternios()
        u[:, k+1] = get_body_vel()

        states[:, k+1] = np.concatenate((h[:, k+1], euler[:, k+1], v[:, k+1], euler_p[:, k+1], omega[:, k+1], quat[:, k+1], u[:, k+1]))
 




    send_reference([0, 0, 0, 0], control_pub, ref_msg)
    

        

    states_data = {"states": states, "label": "states"}
    T_ref_data = {"T_ref": T_ref, "label": "states_ref"}
    t_data = {"t": t, "label": "time"}

    pwd= "/home/bryansgue/Doctoral_Research/Matlab/Identificacion_M100/IdentificacionAlgoritmos/Ident_Full_model_compact"
    

    savemat(os.path.join(pwd, "states_" + str(experiment_number) + ".mat"), states_data)
    savemat(os.path.join(pwd, "T_ref_" + str(experiment_number) + ".mat"), T_ref_data)
    savemat(os.path.join(pwd,"t_"+ str(experiment_number) + ".mat"), t_data)


    return None


if __name__ == '__main__':
    try:
        # Initialization Node
        rospy.init_node("Python_Node",disable_signals=True, anonymous=True)

        # Odometry topic
        odometry_webots = "/dji_sdk/odometry"
        odometry_subscriber = rospy.Subscriber(odometry_webots, Odometry, odometry_call_back)

        # Cmd Vel topic
        velocity_topic = "/m100/velocityControl"
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher(velocity_topic, TwistStamped, queue_size = 10)

        main(velocity_publisher, velocity_message)



    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        send_reference([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        send_reference([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass