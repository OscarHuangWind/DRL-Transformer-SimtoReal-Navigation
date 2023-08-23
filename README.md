# Goal-guided Transformer-enabled Reinforcement Learning for Efficient Autonomous Navigation
:dizzy: **A goal-driven mapless end-to-end autonomous navigation of automated grounded vehicle (AGV) through Transformer-enabled deep reinforcement learning (DRL) algorithm.**

:blue_car: A **car-like mobile robot learns to autonomously navigate to a random goal position only through raw RGB images from one Fisheye camera and goal information in polar coordination system.**

:wrench: Realized in ROS Gazebo simulator with Ubuntu 20.04, ROS noetic, and Pytorch. 

# Preview Simulation
<p align="center">
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_0.gif" width= "45%" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_8.gif" width= "45%" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_2.gif" width= "45%" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_6.gif" width= "45%" />
</p>

# Preview Sim-to-Real Navigation Experiment
:point_right: [GTRL Videos](https://www.youtube.com/watch?v=aqJCHcsj4w0) :point_left:

# Basics
:one: [ROS Noetic](http://wiki.ros.org/noetic/Installation)

:two: [Gazebo](http://classic.gazebosim.org/tutorials?tut=install_from_source&cat=install)

:three: [Pytorch](https://pytorch.org/get-started/locally/)

# User Guidance
## Create a new Virtual environment (conda is suggested).
Specify your own name for the virtual environment, e.g., gtrl:
```
conda create -n gtrl python=3.7
```
## Activate virtual environment.
```
conda activate gtrl
```
## Install Dependencies.
```
pip install numpy tqdm natsort cpprb matplotlib einops squaternion opencv-python
sudo apt install python3-catkin-tools python3-osrf-pycommon
sudo apt-get install ros-noetic-cv-bridge
```
### (Suggested) Optional step for visualizing real-time plotting (reward curve). 
```
conda install spyder==5.2.2
```
## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation.git
```

## Compile the workspace.
```
cd ~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws
```
```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

## Set up the environment variables.
```
export PYTHONPATH=~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts
export GAZEBO_RESOURCE_PATH=~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/launch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/noetic/lib
```
Alternatively, you can select to write these variables to the ~/.bashrc file so that it can be automatically set when opening terminal.

## Source the workspace.
```
source devel/setup.bash
```
## :heavy_exclamation_mark:Important:heavy_exclamation_mark: 
Copy all the files under models folder to your default gazebo models folder.
```
cp -a ~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/models/. ~/.gazebo/models
```
## Revise your system path in main.py file.
```
import sys
sys.path.append('/home/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts')
```
## Time to train and get your GTRL model!!!
```
cd ~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts/SAC
```
Run it in the terminal:
```
python main.py
```
(suggested) Alternatively, if you have already installed spyder, just click the run file button in spyder.

## To kill the program, it is suggested to use following commands.
```
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```
Alternatively, you can add alias of these commands to the ~/.bashrc file:
```
alias k9='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3'
```
And type the alias in the terminal to kill all the process:
```
k9
```

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/framework_final.png" width="70%">
</p>

# Goal-guided Transformer (GoT)
<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/GoalTransformer_final.png" width="80%">
</p>

# Noise-augmented RGB images from fisheye camera
<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/fisheye_final.png" width="60%">
</p>

# AGV and lab environment model in simulation and real world.
<p align="center">
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/gazebo_scout.png" height= "150" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/gazebo_world.png" height= "150" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/AGV.png" height= "150" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/Robotics_Research_Centre.png" height= "150" />
</p>

# Sim-to-Real navigaiton experiment in office environment.
<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/office_environment.png" width="60%">
</p>
