import math
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import board
import adafruit_mcp4728


# Based on: https://www.instructables.com/Autonomous-Lane-Keeping-Car-Using-Raspberry-Pi-and/
# Reference: Team Really Bad Idea https://www.hackster.io/really-bad-idea/autonomous-path-following-car-6c4992 


# Setup the digital-analog converter
i2c = board.I2C()
control = adafruit_mcp4728.MCP4728(i2c, 0x60)

# RPI params
go_forward_default = 36000
go_faster_addition = 2000
dont_move = 31000
left = 10000
right = 55535
go_faster_tick = 0  # Do not change this here. Code will set this value after seeing stop sign
go_faster_tick_delay = 80


"""
# Throttle BBA part, no need for the rpi but can be a reference
throttlePin = "P9_14"
go_forward = 8.15
go_faster_addition = 0.01
go_faster_tick_delay = 80
go_faster_tick = 0  # Do not change this here. Code will set this value after seeing stop sign
dont_move = 7.5

# Steering
steeringPin = "P9_16"
left = 8.8
right = 6.2
"""
# Max number of loops
max_ticks = 2000

# Booleans for handling stop light
passedStopLight = False
atStopLight = False
passedFirstStopSign = False

secondStopLightTick = 0

class Controller():
    def __init__(self):
        self.speed = 32768
        self.turn = 32768
        control.channel_a.value = self.speed
        control.channel_b.value = self.turn

    def set_turn_and_speed(self, speed, turn):
        if speed > 0:
            self.speed = speed

        if turn > 0:
            self.turn = turn
        control.channel_a.value = self.speed
        control.channel_b.value = self.turn

    def set_speed(self, speed):
        if speed > 0:
            if speed < 65535:
                self.speed = int(speed)
                # print("set speed called")
        control.channel_a.value = self.speed

    def set_turn(self, turn):
        if turn > 0:
            if turn < 65535:
                self.turn = int(turn)
                #print("Turn: ", turn)
        control.channel_b.value = self.turn

    def go_faster(self):
        if self.speed < 40000:
            self.speed += go_faster_addition
        if self.speed > 65535:
            self.speed = 65535
        control.channel_a.value = self.speed

    def go_slower(self):
        self.speed -= go_faster_addition
        if self.speed < 0:
            self.speed = 0
        control.channel_a.value = self.speed

    def stop(self):
        self.speed = 32768
        self.turn = 32768
        control.channel_a.value = self.speed
        control.channel_b.value = self.turn

    def go_backwards(self):
        self.speed = 65535 - go_forward_default
        control.channel_a.value = self.speed

    def get_speed(self):
        return self.speed
    def cleanup(self):
        self.speed = 32768
        self.turn = 32768
        control.channel_a.value = self.speed
        control.channel_b.value = self.turn
        print("Jobs finished get some rest y'all")


controller = Controller()

def getRedFloorBoundaries():
    """
    Gets the hsv boundaries and success boundaries indicating if the floor is red
    :return: [[lower color and success boundaries for red floor], [upper color and success boundaries for red floor]]
    """
    return getBoundaries("redboundaries.txt")


def isRedFloorVisible(frame):
    """
    Detects whether or not the floor is red
    :param frame: Image
    :return: [(True is the camera sees a red on the floor, false otherwise), video output]
    """
    # print("Checking for floor stop")
    boundaries = getRedFloorBoundaries()
    return isMostlyColor(frame, boundaries)

def isMostlyColor(image, boundaries):
    """
    Detects whether or not the majority of a color on the screen is a particular color
    :param image:
    :param boundaries: [[color boundaries], [success boundaries]]
    :return: boolean if image satisfies provided boundaries, and an image used for debugging
    """
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # parse out the color boundaries and the success boundaries
    color_boundaries = boundaries[0]
    percentage = boundaries[1]

    lower = np.array(color_boundaries[0])
    upper = np.array(color_boundaries[1])
    mask = cv2.inRange(hsv_img, lower, upper)
    output = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

    # Calculate what percentage of image falls between color boundaries
    percentage_detected = np.count_nonzero(mask) * 100 / np.size(mask)
    # print("percentage_detected " + str(percentage_detected) + " lower " + str(lower) + " upper " + str(upper))
    # If the percentage percentage_detected is between the success boundaries, we return true, otherwise
    # false for result
    result = percentage[0] < percentage_detected <= percentage[1]
    if result:
        # print(percentage_detected)
        pass
    return result, output


def getBoundaries(filename):
    """
    Reads the boundaries from the file filename
    Format:
        [0] lower: [H, S, V, lower percentage for classification of success]
        [1] upper: [H, S, V, upper percentage for classification of success]
    :param filename: file containing boundary information as above
    :return: [[lower color and success boundaries], [upper color and success boundaries]]
    """
    default_lower_percent = 50
    default_upper_percent = 100
    with open(filename, "r") as f:
        boundaries = f.readlines()
        lower_data = [val for val in boundaries[0].split(",")]
        upper_data = [val for val in boundaries[1].split(",")]

        if len(lower_data) >= 4:
            lower_percent = float(lower_data[3])
        else:
            lower_percent = default_lower_percent

        if len(upper_data) >= 4:
            upper_percent = float(upper_data[3])
        else:
            upper_percent = default_upper_percent

        lower = [int(x) for x in lower_data[:3]]
        upper = [int(x) for x in upper_data[:3]]
        boundaries = [lower, upper]
        percentages = [lower_percent, upper_percent]
    return boundaries, percentages


def initialize_car():
    print(
        """Me when I, car.  
        ______
        /|_||_\`.__
        (   _    _ _\ \n
        =`-(_)--(_)-
        """
    )


def stop():
    """
    Stops the car
    :return: none
    """
    controller.set_speed(dont_move)


def go():
    """
    Sends the car forward at a default PWM
    :return: none
    """
    controller.set_speed(go_forward_default)
    #print("go called")

def go_faster():
    """
    Sends the car forward at a faster default PWM
    :return: none
    """
    controller.go_faster()

def go_slower():
    """
    Sends the car forward at a slower PWM
    :return: none
    """
    controller.go_slower()

def go_backwards():
    """
    (Attempts to) send the car backwards
    :return: none
    """
    controller.go_backwards()


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV",hsv)
    lower_blue = np.array([90, 120, 0], dtype="uint8")
    upper_blue = np.array([150, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow("mask",mask)

    # detect edges
    edges = cv2.Canny(mask, 50, 100)
    # cv2.imshow("edges",edges)

    return edges


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array([[
        (0, height),
        (0, height / 2),
        (width, height / 2),
        (width, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, mask)
    # cv2.imshow("roi",cropped_edges)

    return cropped_edges


def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10

    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=5, maxLineGap=150)

    return line_segments


def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("no line segments detected")
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # print("skipping vertical lines (slope = infinity")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines


def make_points(frame, line):
    height, width, _ = frame.shape

    slope, intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape

    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


def plot_pd(p_vals, d_vals, error, show_img=False):
    fig, ax1 = plt.subplots()
    t_ax = np.arange(len(p_vals))
    ax1.plot(t_ax, p_vals, '-', label="P values")
    ax1.plot(t_ax, d_vals, '-', label="D values")
    ax2 = ax1.twinx()
    ax2.plot(t_ax, error, '--r', label="Error")

    ax1.set_xlabel("Frames")
    ax1.set_ylabel("PD Value")
    ax2.set_ylim(-90, 90)
    ax2.set_ylabel("Error Value")

    plt.title("PD Values over time")
    fig.legend()
    fig.tight_layout()
    plt.savefig("pd_plot.png")

    if show_img:
        plt.show()
    plt.clf()


def plot_pwm(speed_pwms, turn_pwms, error, show_img=False):
    fig, ax1 = plt.subplots()
    t_ax = np.arange(len(speed_pwms))
    ax1.plot(t_ax, speed_pwms, '-', label="Speed PWM")
    ax1.plot(t_ax, turn_pwms, '-', label="Steering PWM")
    ax2 = ax1.twinx()
    ax2.plot(t_ax, error, '--r', label="Error")

    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Controller Values")
    ax2.set_ylabel("Error Value")

    plt.title("Controller Values over time")
    fig.legend()
    plt.savefig("Con_plot.png")

    if show_img:
        plt.show()
    plt.clf()

# since this is a car so allow me to do it in WarHammer 40K Mechannicus Adeptus style
# Ring the bell once, activate piston and pumps
initialize_car()

# Ring the bell twice, fire the engine, give birth to life (the video)
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Ring the bell third times, all praise the Omnisiiah (so we sleep(1))
time.sleep(1)

# PD variables
kp = 0.095
kd = kp * 0.1
lastTime = 0
lastError = 0

# counter for number of ticks
counter = 0

# start the engines
go()

# arrays for making the final graphs
p_vals = []
d_vals = []
err_vals = []
speed_pwm = []
steer_pwm = []
current_speed = go_forward_default

stopSignCheck = 1
sightDebug = False
isStopSignBool = False

# Too much work I need a rest
while counter < max_ticks:
    ret, original_frame = video.read()
    frame = cv2.resize(original_frame, (160, 120))
    if sightDebug:
        cv2.imshow("Resized Frame", frame)

    # reading the encoder data and changing the speed   
    time_diff = 7

    # This need attention: we will see where is this param
    with open("/sys/module/speed_driver_no_tree/parameters/elapsed", "r") as filetoread:
        time_diff = int(filetoread.read())

#    print("Time diff", time_diff)

    # if time_diff > 2000:
    #     time_diff = 200
    # Encoder time me when I check the encoder
    if time_diff == 0: # error detection
        go()
    elif time_diff >= 30: # time_diff > 120
        go_faster()
    elif time_diff <= 30: # 7 < time_diff < 110
        go_slower()

    # check for stop sign/traffic light every couple ticks
    if ((counter + 1) % stopSignCheck) == 0:
        # check for the first stop sign
        if not passedFirstStopSign:
            isStopSignBool, floorSight = isRedFloorVisible(frame)
            if sightDebug:
                cv2.imshow("floorSight", floorSight)
            if isStopSignBool:
                print("detected first stop sign, stopping")
                stop()
                time.sleep(2)
                passedFirstStopSign = True
                # this is used to not check for the second stop sign until many frames later
                secondStopSignTick = counter + 200
                # now check for stop sign less frequently
                stopSignCheck = 3
                # add a delay to calling go faster
                go_faster_tick = counter + go_faster_tick_delay
                print("first stop finished!")
                go_faster()
                go_faster()
        # check for the second stop sign
        elif passedFirstStopSign and counter > secondStopSignTick:
            isStop2SignBool, _ = isRedFloorVisible(frame)
            if isStop2SignBool:
                # last stop sign detected, exits while loop
                print("detected second stop sign, stopping")
                stop()
                break

    # makes car go faster, helps it have enough speed to get to the end of the course
    if isStopSignBool and counter == go_faster_tick:
        print("Going FASTER")
        go_faster()

    # process the frame to determine the desired steering angle
    # cv2.imshow("original",frame)
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    # heading_image = display_heading_line(lane_lines_image,steering_angle)
    # cv2.imshow("heading line",heading_image)

    # calculate changes for PD
    now = time.time()
    dt = now - lastTime
    if sightDebug:
        cv2.imshow("Cropped sight", roi)
    deviation = steering_angle - 90

    # PD Code
    error = -deviation
    base_turn = 32768
    proportional = kp * error
    derivative = kd * (error - lastError) / dt

    # take values for graphs
    p_vals.append(proportional)
    d_vals.append(derivative)
    err_vals.append(error)

    # determine actual turn to do
    turn_amt = base_turn - (proportional + derivative) * 50000

    # caps turns to make PWM values
    if 25206 < turn_amt < 40329:
        turn_amt = 32768
    elif turn_amt < left:
        turn_amt = left
    elif turn_amt > right:
        turn_amt = right
    else:
        turn_amt = turn_amt

    # turn!
    controller.set_turn(turn_amt)
    current_speed = controller.get_speed()
    #print("turn_amt", turn_amt)
    #print("speed", current_speed)

    # take values for graphs
    steer_pwm.append(turn_amt)
    speed_pwm.append(current_speed)

    # update PD values for next loop
    lastError = error
    lastTime = time.time()

    key = cv2.waitKey(1)
    if key == 27:
        break

    counter += 1

'''
 _______
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%               %%%%%              %%%%%   %%%%%%%%   %%%%%           %%%%%%%%%%%   %%%%%%   %%%
 %%%%%%%%   %%%%%%%%%%%%%%%%%%%%%%%   %%%%%%%%%%   %%%%%%%%   %%%%%   %%%%%%%   %%%%%%%%%%   %%%%   %%%%
 %%%%%%%%   %%%%%%%%%%%%%%%%%%%%%%%   %%%%%%%%%%   %%%%%%%%   %%%%%   %%%%%%%%%   %%%%%%%%%   %%   %%%%%
 %%%%%%%%               %%%%%%%%%%%   %%%%%%%%%%   %%%%%%%%   %%%%%   %%%%%%%%%%   %%%%%%%%%      %%%%%%
 %%%%%%%%%%%%%%%%%%%   %%%%%%%%%%%%   %%%%%%%%%%   %%%%%%%%   %%%%%   %%%%%%%%%   %%%%%%%%%%%   %%%%%%%%
 %%%%%%%%%%%%%%%%%%%   %%%%%%%%%%%%   %%%%%%%%%%   %%%%%%%%   %%%%%   %%%%%%%   %%%%%%%%%%%%%   %%%%%%%%
 %%%%%%%%              %%%%%%%%%%%%   %%%%%%%%%%              %%%%%           %%%%%%%%%%%%%%%   %%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

# clean up resources
video.release()
cv2.destroyAllWindows()
controller.cleanup()

plot_pd(p_vals, d_vals, err_vals, True)
plot_pwm(speed_pwm, steer_pwm, err_vals, True)
