# GoT-GTRL
### :page_with_curl: Goal-guided Transformer-enabled Reinforcement Learning for Efficient Autonomous Navigation
### [[**Published Paper**]](https://ieeexplore.ieee.org/document/10254445) | [[**arXiv**]](https://arxiv.org/abs/2301.00362) | [[**BiliBili**]](https://www.bilibili.com/video/BV1Mj41147Md/) | [[**Youtube**]](https://www.youtube.com/watch?v=aqJCHcsj4w0&t=1s)

:dizzy: **A goal-driven mapless end-to-end autonomous navigation of unmanned grounded vehicle (UGV) realized through Transformer-enabled deep reinforcement learning (DRL) algorithm.**

:blue_car: A **car-like mobile robot learns to autonomously navigate to a random goal position only through raw RGB images from one Fisheye camera and goal information in polar coordination system.**

:wrench: Realized in ROS Gazebo simulator with Ubuntu 20.04, ROS noetic, and Pytorch. 

# Citation
If you find this repository useful for your research, please consider starring :star: our repo and citing our paper.
```
@article{huang2024goal,
  title={Goal-Guided Transformer-Enabled Reinforcement Learning for Efficient Autonomous Navigation},
  author={Huang, Wenhui and Zhou, Yanxin and He, Xiangkun and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={25},
  number={2},
  pages={1832--1845},
  year={2024},
  publisher={IEEE}
}
```

# Preview Simulation 
Click the gif to zoom in :mag_right:
<p align="center">
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_0.gif" width= "45%" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_8.gif" width= "45%" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_2.gif" width= "45%" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/demo_6.gif" width= "45%" />
</p>

# Video: Sim-to-Real Experiment :arrow_lower_left:
:point_right: [<ins>GTRL Sim-to-Real Navigation Experiment Video<ins>](https://www.youtube.com/watch?v=aqJCHcsj4w0) :point_left:

# Basic Dependency Installation
:one: [ROS Noetic](http://wiki.ros.org/noetic/Installation)

:two: [Gazebo](https://classic.gazebosim.org/tutorials?tut=install_ubuntu)

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
pip install numpy tqdm natsort cpprb matplotlib einops squaternion opencv-python rospkg rosnumpy yaml
sudo apt install python3-catkin-tools python3-osrf-pycommon
sudo apt-get install ros-noetic-cv-bridge
```
### Optional step for visualizing real-time plotting (reward curve) with Spyder. 
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
export GAZEBO_RESOURCE_PATH=~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/launch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/noetic/lib
```
Alternatively, you can select to write these variables to the ~/.bashrc file so that it can be automatically set when opening terminal.

## Source the workspace.
```
source devel/setup.bash
```
## Important!
Copy all the files under models folder to your default gazebo models folder.
```
cp -a ~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/models/. ~/.gazebo/models
```
## Revise your system path in main.py and env_lab.py (gtrl/scripts/Environments/env_lab.py) file.
main.py
```
import sys
sys.path.append('/home/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts')
```
env_lab.py (line 129)
```
fullpath = os.path.join('/home/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/drl_navigation/launch', launchfile)
```
## Time to train and get your GTRL model!!!
```
cd ~/$your workspace/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts/SAC
```
Run it in the terminal:
```
python main.py
```
(Optional) Alternatively, if you have already installed spyder, just click the run file button in spyder.

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
