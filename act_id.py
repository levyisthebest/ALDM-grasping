import pyrealsense2 as rs
import numpy as np
import time
import cv2
import json
from real_run_DT import load_agent


def get_image_from_cam():

    deviceSerials = ['215122255902', '151222075610']

    conf1 = rs.config()
    conf1.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    conf1.enable_device(deviceSerials[0])
    pipe1 = rs.pipeline()

    conf2 = rs.config()
    conf2.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    conf2.enable_device(deviceSerials[1])
    pipe2 = rs.pipeline()

    pipelineProfile1 = pipe1.start(conf1)
    pipelineProfile2 = pipe2.start(conf2)

    # print(pipelineProfile1.get_device().first_depth_sensor()) 

    agent = load_agent('./checkpoints/RAL/ObjectDet_Swin_Ti_12k') # netDet_20_12k_block  RAL/ObjectDet_Swin_Ti_12k
    timeout = time.time() + 360
    bboxs = []
    arm_down = 0
    count = 0

    t = 'yblock'

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipe1.wait_for_frames()
            frames2 = pipe2.wait_for_frames()

            # image_timestamp = time.time()
            color_frame = frames.get_color_frame()
            color_image_base = np.asanyarray(color_frame.get_data())

            color_frame2 = frames2.get_color_frame()
            color_image_arm = np.asanyarray(color_frame2.get_data())

            color_images = np.vstack((color_image_arm, color_image_base))

            state = np.dstack((color_image_arm, color_image_base)).transpose(2,0,1)
            bbox = agent.sample(state/255.0)

            bboxs.append(bbox)
            if count%2==0:
                s = time.time()
                bbox = np.mean(bboxs, axis=0)
                act_id, arm_down, dist, error = agent.get_action_from_obs(bbox, arm_down)
                print(count, '#######################')
                print('action: {}\narm_down: {}\ndist: {:.04f}\nerror{}'.format(act_id, arm_down, dist, np.round(error, 4)))
                res = [act_id, arm_down, count]
                np.save('actID', res)
                end = time.time()
                print('ffffffffffff', end - s)
                bboxs = []
                
            count+=1

            # Show images

            start = time.time()

            box1 = np.rint([bbox[0]*1280, bbox[1]*720, bbox[2]*1280, bbox[3]*720])
            box2 = np.rint([bbox[4]*1280, bbox[5]*720, bbox[6]*1280, bbox[7]*720])
            # print(count, ':', (box1[0]+box1[2])/2,(box1[1]+box1[3])/2,(box2[0]+box2[2])/2,(box2[1]+box2[3])/2)

            color_images = cv2.cvtColor(color_images, cv2.COLOR_BGR2RGB)
            color_images = cv2.rectangle(color_images,(round(box1[0]), round(box1[1])),(round(box1[2]), round(box1[3])),(0,0,255),2)
            color_images = cv2.rectangle(color_images,(round(box2[0]), round(box2[1])+720),(round(box2[2]), round(box2[3])+720),(0,0,255),2)
            
            cv2.namedWindow('RealSense 1', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RealSense 1', 600, 600) 
            cv2.imshow('RealSense 1', color_images)

            end = time.time()

            print('time', end -start)

            # cv2.imwrite('stream/ral/{}/{:03d}.jpg'.format(t,count), color_images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27 or time.time() > timeout or count >=500:
                cv2.destroyAllWindows()
                break

            # if time.time() > timeout:
            #     break

    finally:
        # Stop streaming
        pipe1.stop()
            
    return count

if __name__=='__main__':

    count = get_image_from_cam()