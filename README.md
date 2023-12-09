# Studious-RC-Car
An autonomous lane-keeping RC car on Raspberry Pi 4

Story
This project is about a OpenCV-based autonomous car that is able to auto-follow a given path and stop at given stop signs. The path is composed of blue tapes and the stop signs are red papers.

The idea is based on User raja_961, “Autonomous Lane-Keeping Car Using RaspberryPi and OpenCV”. Instructables. (URL: https://www.instructables.com/Autonomous-Lane-Keeping-Car-Using-Raspberry-Pi-and/).

The project main code is originated from EthanDP, "424-lane-keeping-project" (URL: https://github.com/EthanDP/424-lane-keeping-project), which is a project in Hackster by Team There are four of us, "Autonomous RC Car" (URL: https://www.hackster.io/there-are-four-of-us/autonomous-rc-car-d71671).

The Driver is modified from Johannes4Linux, "Linux_Driver_Tutorial", (URL: https://github.com/Johannes4Linux/Linux_Driver_Tutorial/blob/main/11_gpio_irq/gpio_irq.c)

Hardware
For hardware part, we connected the speed encoder to the back of the car. For the main body of the car, we used a hollow box to store all electrical components including the controller PCB, the RPi4, as well as the DAC module. On the box, we left holes for wires to go through, so that we used another box to mount the webcam and connect the webcam to the RPi4.

Parameter
We used 160 x 120 camera resolution to achieve both a higher accuracy of lane detection and a faster computational speed. Through experiments, we concluded that setting the proportional gain to 0.095 and the derivative gain to 0.0095 gives the best result. To change the code by Team There are four of us which runs on BeagleBone Black to be compatible with our Raspberry Pi 4, we set the threshold voltage to 50000 times the original value.

