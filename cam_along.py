import pyrealsense2 as rs
import numpy as np
import time
import cv2
import json
from real_run_DT import load_agent
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt 

def get_image_from_cam():

    deviceSerials = ['215122255902', '151222075610']

    conf1 = rs.config()
    conf1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    conf1.enable_device(deviceSerials[0])
    pipe1 = rs.pipeline()

    conf2 = rs.config()
    conf2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    conf2.enable_device(deviceSerials[1])
    pipe2 = rs.pipeline()

    pipelineProfile1 = pipe1.start(conf1)
    pipelineProfile2 = pipe2.start(conf2)

    # print(pipelineProfile1.get_device().first_depth_sensor()) 
    model = YOLO('./checkpoints/IROS/best.pt')
    #agent = load_agent('./checkpoints/RAL/netDet_20_0k_block.pth') # netG_B2A_withdet_10_12-12k  netDet_20_12k_block
    timeout = time.time() + 3000
    count = 0

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
            np.save('./color_image_arm.npy',color_image_arm)
            Result_arm = model.predict(color_image_arm,max_det=1,conf=0.01)
            Result_base = model.predict(color_image_base,max_det=1,conf=0.01)
            if(len(Result_arm[0].boxes.xyxy)>0 and len(Result_base[0].boxes.xyxy)>0):
                bbox_arm = Result_arm[0].boxes.xyxy[0].cpu().numpy().astype(int)
                print('######################################')
                print(bbox_arm)
                bbox_base = Result_base[0].boxes.xyxy[0].cpu().numpy().astype(int)
                print('######################################')
                print(bbox_base)
                
                #print(color_image_arm.dtype)
                #print(np.max(color_image_arm))
                #image = Image.fromarray(color_image_arm)
                #image.save('./2.png')
                color_images = np.vstack((color_image_arm, color_image_base))

                #state = np.dstack((color_image_arm, color_image_base)).transpose(2,0,1)

            # bbox = agent.sample(state/255.0)

                box1 = np.rint([bbox_arm[0], bbox_arm[1], bbox_arm[2], bbox_arm[3]])
                box2 = np.rint([bbox_base[0], bbox_base[1], bbox_base[2], bbox_base[3]])
                # print(count, ':', (box1[0]+box1[2])/2,(box1[1]+box1[3])/2,(box2[0]+box2[2])/2,(box2[1]+box2[3])/2)
                print(count)

                # Show images
                # time.sleep(0.1)

                #color_images = cv2.cvtColor(color_images, cv2.COLOR_BGR2RGB)
                #color_image_arm = cv2.cvtColor(color_image_arm, cv2.COLOR_BGR2RGB)
                #color_image_base = cv2.cvtColor(color_image_base, cv2.COLOR_BGR2RGB)
                color_images = cv2.rectangle(color_images,(round(box1[0]), round(box1[1])),(round(box1[2]), round(box1[3])),(0,255,0),2)
                color_images = cv2.rectangle(color_images,(round(box2[0]), round(box2[1])+720),(round(box2[2]), round(box2[3])+720),(0,0,255),2)

                t = 'red'
                # cv2.imwrite('stream/new_objs/{}/{:03d}.jpg'.format(t,count), color_images)
                #color_image_arm = cv2.rectangle(color_image_arm,(round(box1[0]), round(box1[1])),(round(box1[2]), round(box1[3])),(0,255,0),2)
                cv2.imwrite('stream/ral/{}/{:03d}_l.jpg'.format(t,count), color_image_arm)
                # cv2.imwrite('stream/new_objs/complex/{}/{:03d}_g.jpg'.format(t,count), color_image_base)

                count+=1
                # if count%160==0:
                #     time.sleep(25)
                cv2.namedWindow('RealSense 1', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense 1', 600, 600) 
                cv2.imshow('RealSense 1', color_images)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27 or time.time() > timeout or count >= 2000:
                    cv2.destroyAllWindows()
                    break

                # if time.time() > timeout:
                #     break

    finally:
        # Stop streaming
        pipe1.stop()
            
    print(count)
    return color_images

if __name__=='__main__':

    color_images = get_image_from_cam()
    print('shape:', color_images.shape)
