from flask import Flask, request
# from real_run_DT import load_agent
# from record import get_act_from_cam
import numpy as np
import time

# global agent

app = Flask(__name__)

@app.route("/")
def hello_world():
    
    # isdown = int(request.args.get('isdown'))
    # step = int(request.args.get('step'))

    # print('step:', step, '#################')

    time.sleep(0.01)

    # action_id, arm_down = get_act_from_cam(agent, isdown)
    start = time.time()
    actID = np.load('actID.npy')
    action_id = actID[0]
    arm_down = actID[1]
    count = actID[2]
    end = time.time()
    print('#########time#########', end - start)

    start = time.time()
    print('count:', count)
    print('arm_down:', arm_down)
    print('action_id:', action_id)
    end = time.time()
    print('#########time22222#########', end - start)
    
    return "{}@{}".format(action_id, arm_down)

if __name__=='__main__':

    # agent = load_agent('./checkpoints/DT-CycleGAN_20.pth')
    app.run(host="0.0.0.0", port=5000, debug=True) # how to input parameter 'action'