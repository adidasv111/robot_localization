/*
 * Copyright (c) 2014, 2015, 2016, Charles River Analytics, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "robot_localization/ros_filter_types.h"
#include <ros/ros.h>

#include "geometry_msgs/PoseWithCovarianceStamped.h"

// TODO (adi.vardi@dorabot.com): Currently this node dies after a short while. one reason could be the short-time declaration of NodeHandle in the callback.
// A good solution could be to replace the callback with a lambda, which can access the NodeHandle (and probably service/topic) decalred in the main().
// Then we can remove the remmaping from set_pose to optical_initial_pose in the launch file.

bool got_init_pose = false;
void initial_pose_callback(const geometry_msgs::PoseWithCovarianceStamped& msg)
{
  // std::cout << "init_pose callback!" << std::endl;
  if (!got_init_pose)
  {
    // using service
    ros::NodeHandle n;
    ros::ServiceClient client = n.serviceClient<robot_localization::SetPose>("set_pose");
    robot_localization::SetPose srv;
    srv.request.pose = msg;
    if (client.call(srv))
    {
    got_init_pose = true;
    }

    // using topic
    // ros::NodeHandle n;
    // ros::Publisher pub = n.advertise<geometry_msgs::PoseWithCovarianceStamped>("/set_pose", 1);
    // geometry_msgs::PoseWithCovarianceStamped init_pose = msg;
    // pub.publish(init_pose);
    // got_init_pose = true;
  }
  usleep(10);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ekf_navigation_node");

  RobotLocalization::RosEkf ekf;

  // subsribe for optical_raw
  ros::NodeHandle n;
  ros::Subscriber init_pose_sub = n.subscribe("/optical_raw", 1, initial_pose_callback);
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // main run method
  ekf.run();

  return EXIT_SUCCESS;
}
