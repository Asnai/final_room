"""This is the script for prediction of 
events      -   light, occupancy
transition  -   aircon status
Data are retrieved and send to
get live data       -   from firebase
send predicted data -   to firebase
Author:         isl connect team
organisation:   ISL  
"""

import numpy as np
import pickle
import pyrebase
from filterpy.kalman import KalmanFilter
from progress.bar import Bar
from pyfcm import FCMNotification
from sklearn import preprocessing
from termcolor import colored

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import getpass


def mostFrequent(arr, n):
    # Sort the array
    arr.sort()
    # find the max frequency using
    # linear traversal
    max_count = 1
    res = arr[0]
    curr_count = 1
    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count += 1
        else:
            if (curr_count > max_count):
                max_count = curr_count
                res = arr[i - 1]
            curr_count = 1

    # If last element is most frequent
    if (curr_count > max_count):
        max_count = curr_count
        res = arr[n - 1]
    return res


def send_notification():
    push_service = FCMNotification(
        api_key="AAAAYZeGLRQ:APA91bFxUFSa1jjVBdx3u6O3hnG5FOX1vb4-AdHtVWmFw5Uz_Z2YqhJbn09YCl05WJn-2b3LNU2Oas5N_pdvfbGdIYOL79t7ogj_SOJ_X86RsBlKJZ3kjEfAg3kodWUr5_Vt1QiIhT-2vAKC1tGGSI7glojgXA1raQ")
    registration_id = "dqKCetGoaNw:APA91bFbr8khPvPk83Np2boHlWXxDDGD7hDj1gvma7mahM41CFMJUuoHMv4Bi5NlfDuhk0lxXBc67RuXJu2RpVAzPkNnxrp89xPpYWI7IAcn9qBX7-0k6C1V0LH0lo0yzP4hQHu4L9IhpVzA3gJF0ii0DhLJdkDfSw"
    message_title = "Sensor Status"
    push_service.notify_single_device(
        registration_id=registration_id, message_title=message_title, message_body=message_body, sound="Default")


def init_firebase():
    config = {
        "apiKey": "AIzaSyCxWGzzmEspAkwe9VyCHGb7mDk2g7HxyS4",
        "authDomain": "smart-connect-4c060.firebaseapp.com",
        "databaseURL": "https://smart-connect-4c060.firebaseio.com",
        "storageBucket": "smart-connect-4c060.appspot.com"
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    return db


def init_kalman_filter(initValue):
    my_filter = KalmanFilter(dim_x=2, dim_z=1)
    # initial state (location and velocity)
    my_filter.x = np.array([[initValue], [0.]])
    my_filter.F = np.array([[1., 1.], [0., 1.]])    # state transition matrix
    my_filter.H = np.array([[1., 0.]])              # Measurement function
    my_filter.P = np.array([[1, 0.], [0., 1]])      # covariance matrix
    my_filter.R = np.array([[1000.]])               # Measurement noise
    my_filter.Q = np.array([[1, 1], [1, 1]])        # process noise
    return my_filter


def get_logistic_regression(data, mid_value):
    # output = 1 / (1 + np.exp(-1 * data + self.b1 * self.b0))
    output = 1 / (1 + np.exp(-1 * data + mid_value))
    return output


def get_logistic_regression_alt(data):
    new_array = []
    mean = np.mean(data)
    for d in data:
        output = 1 / (1 + np.exp(-1 * d + mean))
        new_array.append(output)
    return new_array


def get_fourier_transform(data):
    return np.fft.fft(data).real


def stream_handler(message):
    global temp_array
    global humid_array
    global aircon_status
    global noti_sent
    global initState
    global prediction_state
    global light
    global sound
    global thermopile
    global motion
    global occupancy_status
    global anomaly_status
    global light_status
    global email_activator
    global email
    global password

    email_activator += 1
    prediction_state += 1
    ac_activator = 0

    current_datetime = message["data"]["date_time"]
    current_light = message["data"]["light_value"]
    current_sound = message["data"]["sound_value"]
    current_temp = message["data"]["temperature_value"]
    current_humid = message["data"]["humidity_value"]
    current_motion = message["data"]["motion_value"]
    current_thermopile = message["data"]["thermopile_value"]

    tempKalman.predict()
    thermopileKalman.predict()
    humidKalman.predict()
    soundKalman.predict()
    motionKalman.predict()
    kalman_temp = tempKalman.x[0][0]
    kalman_thermopile = thermopileKalman.x[0][0]
    kalman_humid = humidKalman.x[0][0]
    kalman_sound = soundKalman.x[0][0]
    soundKalman.R = np.array([[500.]])
    kalman_motion = motionKalman.x[0][0]
    tempKalman.update(current_temp)
    thermopileKalman.update(current_thermopile)
    humidKalman.update(current_humid)
    soundKalman.update(current_sound)
    motionKalman.update(current_motion)

    logit_temp = get_logistic_regression(kalman_temp, 25)
    logit_humid = get_logistic_regression(kalman_humid, 50)
    logit_sound = get_logistic_regression(kalman_sound, 1)
    logit_motion = get_logistic_regression(kalman_motion, 0)

    # aircon status prediction
    if len(temp_array) != 1200 and len(humid_array) != 1200:
        # progress_bar.next()
        fourier_temp = []
        fourier_humid = []
        temp_array.append(kalman_temp)
        humid_array.append(kalman_humid)
    else:
        ac_activator = 1
        present = temp_array[-20:]
        past = temp_array[:20]
        result = []

        for i in range(20):
            if (present[i] - past[i]) > 1.0:
                result.append(0)
            elif (present[i] - past[i]) < -1.0:
                result.append(1)
            else:
                result.append(-1)
        
        maximum = mostFrequent(result,len(result))
        if maximum == 0:
            aircon_status = 'OFF'
            print ('ac is off')
        elif maximum == 1:
            aircon_status = 'ON'
            print ('ac is on')

        temp_array[:-1] = temp_array[1:]
        temp_array[-1] = kalman_temp
        humid_array[:-1] = humid_array[1:]
        humid_array[-1] = kalman_humid
        temp_array = get_logistic_regression_alt(temp_array)
        humid_array = get_logistic_regression_alt(humid_array)
        fourier_temp = get_fourier_transform(temp_array)
        fourier_humid = get_fourier_transform(humid_array)
        # data_arr = []
        # for idx, temp in enumerate(fourier_temp):
        #     data_arr.append(temp)
        #     data_arr.append(fourier_humid[idx])
        # ac_result = aircon_model.predict([data_arr])

        # print("ac_result", ac_result)
        # if ac_result[0] == 0:
        #     aircon_status = 'OFF'
        # else:
        #     aircon_status = 'ON'

    fourier_temp_str = str(fourier_temp)
    fourier_humid_str = str(fourier_humid)

    # light status prediction
    # occupancy status prediction
    if prediction_state == 6:  # predict after interval of 5
        avg_light = sum(light)/len(light)
        light_result = light_model.predict(avg_light)
        light_status = str(light_result)
        light = []
        avg_sound = sum(sound)/len(sound)
        print(sound)
        avg_thermopile = sum(thermopile)/len(thermopile)
        max_motion = mostFrequent(motion, len(motion))
        x1 = np.append(max_motion, avg_sound)
        x2 = np.append(kalman_temp, avg_thermopile)
        x = np.append(x1, x2)
        occupancy_result = occupancy_model.predict([x])
        occupancy_status = str(occupancy_result)
        sound = []
        thermopile = []
        motion = []
        prediction_state = 1
        print('update predictions' + str(email_activator))


        if email_activator > 600:  # email activation with 10 mins interval
            if (occupancy_status == "[0.]" and light_status == "[1.]") or (occupancy_status == "[0.]" and aircon_status == "ON"):
                anomaly_status = "[1.]"
                toaddr = ["narangasees@gmail.com", "u5815228@au.edu"]  # receiver email address
                msg = MIMEMultipart()
                msg['From'] = email
                msg['To'] = ", ".join(toaddr)
                msg['Subject'] = "Alert From Room Sensor"

                body = "Lights or ac in Room XXX might be left on even though the room is not being used."
                body = body + " You should go check it out. \n \n Regards, \n Your Hardworking Sensors."
                msg.attach(MIMEText(body, 'plain'))

                # gmail smtp server
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                # login with senders email address and password
                server.login(email, password)
                text = msg.as_string()
                server.sendmail(email, toaddr, text)
                server.quit()
                email_activator = 0
            else:
                anomaly_status = "[0.]"

        if email_activator > 600:
            email_activator = 601

    light.append(current_light)
    sound.append(kalman_sound)
    motion.append(current_motion)
    thermopile.append(kalman_thermopile)

    occupancy_msg = ""
    light_msg = ""
    if occupancy_status == "[1.]":
        occupancy_msg = "occupied"
    else:
        occupancy_msg = "empty   "

    if light_status == "[0.]":
        light_msg = "off"
    else:
        light_msg = "on"

    # push notification
    # if light_status == "[0.]" and occupancy_status == "[1.]" and noti_sent == False:
    #     noti_sent = True
    #     message_body = ("From raspi-2: Thief suspected here")
    #     result = push_service.notify_single_device(registration_id = registration_id, message_title=message_title, message_body=message_body, sound="Default")
    # else:
    #     noti_sent = False

    print("occupancy status: {} , light status: {}, kalman sound: {}".format(colored(occupancy_msg, "green"),
                                                                             colored(light_msg, "blue"), colored(kalman_sound, "yellow")))

    json_data = {
        'datetime': current_datetime,
        'light_raw': current_light,
        'light_status': light_status,
        'temp_raw': current_temp,
        'temp_kalman': kalman_temp,
        'temp_logit': logit_temp,
        'humid_raw': current_humid,
        'humid_kalman': kalman_humid,
        'humid_logit': logit_humid,
        'aircon_status': aircon_status,
        'sound_raw': current_sound,
        'sound_kalman': kalman_sound,
        'sound_logit': logit_sound,
        'motion_raw': current_motion,
        'motion_kalman': kalman_motion,
        'motion_logit': logit_motion,
        'occupancy_status': occupancy_status,
        'anomaly_status' : anomaly_status
    }
    db.child("process").set(json_data)

    fourier_dict = {
        'temperature': fourier_temp_str,
        'humidity': fourier_humid_str
    }
    # print(fourier_dict)
    db.child("fourier").set(fourier_dict)
    # db.child("fourier").child("temp").set(fourier_temp)
    # db.child("fourier").child("humid").set(fourier_humid)


if __name__ == "__main__":
    temp_array = []
    humid_array = []
    aircon_status = 'OFF'
    noti_sent = False
    initState = 0
    prediction_state = 0
    occupancy_status = "[0.]"
    light_status = "[0.]"
    anomaly_status = "[0.]"
    motion = []
    light = []
    sound = []
    thermopile = []
    email_activator = 0

    email = input('Email: ')  # get email
    password = getpass.getpass('Password:')  # get password

    db = init_firebase()
    tempKalman = init_kalman_filter(25)
    thermopileKalman = init_kalman_filter(25)
    humidKalman = init_kalman_filter(60)
    soundKalman = init_kalman_filter(1)
    motionKalman = init_kalman_filter(0)

    light_model = pickle.load(open('./models/light_v2.sav', 'rb'))
    aircon_model = pickle.load(open("./models/aircon_svm.sav", "rb"))
    occupancy_model = pickle.load(open("./models/occupancy-Nov.sav", "rb"))
    # progress_bar = Bar("Processing", max= 1200)
    # init_progress_bar = Bar("Initializing", max = 10)
    occupancy_stream = db.child("streaming-raspi-2").stream(stream_handler)
