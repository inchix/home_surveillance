# Home Surveillance with Facial Recognition 

Smart security is the future, and with the help of the open source community and technology available today, an affordable intelligent video analytics system is within our reach. This application is a low-cost, adaptive and extensible surveillance system focused on identifying and alerting for potential home intruders. It can integrate into an existing alarm system and provides customizable alerts for the user. It can process several IP cameras and can distinguish between someone who is in the face database and someone who is not (a potential intruder).

---

[![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/revert-29-master/system/testing/implementation/dashboard.png)](#features)

## System Overview ##

### What's inside? ###

The main system components include a dedicated system server which performs all the central processing and web-based communication and a Raspberry PI which hosts the alarm control interface.  


[![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/revert-29-master/system/testing/implementation/designOverview-2.png)](#features)

### How does it work? ###

The SurveillanceSystem object is the heart of the system. It can process several IPCameras and monitors the system's alerts. A FaceRecogniser object provides functions for training a linear SVM classifier using the face database and includes all the functions necessary to perform face recognition using Openface's pre-trained neural network (thank you Brandon Amos!!). The IPcamera object streams frames directly from an IP camera and makes them available for processing, and streaming to the web client. Each IPCamera has its own MotionDetector and FaceDetector object, which are used by other subsequent processes to perform face recognition and person tracking. The FlaskSocketIO object streams jpeg frames (mjpeg) to the client and transfers JSON data using HTTP POST requests and web sockets. Finally, the flask object on the Raspberry PI simply controls a GPIO interface which can be directly connected to an existing wired alarm panel.
 
 [![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/revert-29-master/system/testing/implementation/finalSystemImplementation.png)](#features)

### How do I setup the network? ###

How the network is setup is really up to you. I used a PoE switch to connect all my IP cameras to the network, and you can stream from cameras that are directly connected to an NVR.

 [![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/revert-29-master/system/testing/implementation/testingEnvironment.png)](#features)


## Facial Recognition Accuracy ##

The graph below shows the recognition accuracy of identifying known and unknown people with the use of an unknown class in the classifier and an unknown confidence threshold. Currently, Openface has an accuracy of 0.9292 ± 0.0134 on the LFW benchmark, and although benchmarks are great for comparing the accuracy of different techniques and algorithms, they do not model a real world surveillance environment. The tests conducted were taken in a home surveillance scenario with two different IP cameras in an indoor and outdoor environment at different times of the day. A total of 15 people were recorded and captured to train the classifier. Face images were also taken from both the LFW database as well as the FEI database, to test the recognition accuracy of identifying unknown people and create the unknown class in the classifier. 


 [![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/revert-29-master/system/testing/implementation/RecognitionAccuracy.png)](#features)
 

At an unknown confidence threshold of 20, the recognition accuracy of identifying an unknown person is 81.25%, while the accuracy of identifying a known person is 75.52%. This produces a final combined system recognition accuracy of 78.39%. 
 
## System Processing Capability ##

The systems ability to process several cameras simultaneously in real time with different resolutions is shown in the graph below. These tests were conducted on a 2011 Mac Book Pro running Yosemite. 

 [![solarized dualmode](https://raw.githubusercontent.com/BrandonJoffe/home_surveillance/revert-29-master/system/testing/implementation/processingCapability.png)](#features)

By default, the SurveillanceSystem object resizes frames to a ratio where the height is always 640 pixels. This was chosen as it produced the best results with regards to its effects on processing and face recognition accuracy. Although the graph shows the ability of the system to process up to 9 cameras in real time using a resolution of 640x480, it cannot stream 9 cameras to the web client simultaneously with the approach currently being used. During testing up to 6 cameras were able to stream in real time, but this was not always the case. The most consistent real-time streaming included the use of only three cameras.

# License
---

Copyright 2016, Brandon Joffe, All rights reserved.
Copyright 2020, domcross

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

- http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# References
---

- Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/
- Flask Web Server GPIO - http://mattrichardson.com/Raspberry-Pi-Flask/
- Openface Project - https://cmusatyalab.github.io/openface/
- Flask Websockets - http://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent

 


 
