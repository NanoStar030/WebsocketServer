import asyncio
import logging
import websockets
from ultralytics import YOLO
from flask import Flask, request, render_template, send_file

Hololens_Client = None
app = Flask(__name__, template_folder="public")
log = logging.getLogger("werkzeug")
log.setLevel(logging.CRITICAL) 
## Services for website ##
@app.route("/home")
def home():
    return render_template("home.html")
@app.route("/picture")
def get_picture():
    picture_path = "public/test.jpg"
    return send_file(picture_path, mimetype="image/jpeg")
@app.route("/ReceivePoint", methods=["POST"])
def receivePoint():
    color = request.form["Color"]
    xpos = request.form["Xpos"]
    ypos = request.form["Ypos"]
    SendHintPoint(color, xpos, ypos)
    return {"message": "Point received successfully"}

async def handle_client(websocket):
    try:
        print("Client connected from {0}".format(websocket.remote_address))
        async for message in websocket:
            if isinstance(message, str):
                await ReceivedStrData(websocket, message)
            if isinstance(message, bytes):
                ReceivedByteData(message)
        print("Client disconnected from {0}".format(websocket.remote_address))
    except:
        print("Client connected error")
        
async def ReceivedStrData(websocket, data):
    # print(data)
    # data format: 
    # Message:string
    # UpdateClient:hololens/website
    # YoloDetect:-1/0/1/2/3...
    command, value = data.split(':')
    if command == "UpdateClient":
        if value == "Hololens":
            global Hololens_Client 
            Hololens_Client = websocket
            print("{0} is new Hololens Client now".format(Hololens_Client.remote_address))
            await Hololens_Client.send("Message:Update Successfully")
    elif command == "YoloDetect":
        message = YoloDetect(value)
        await SendYoloPoints(message)
    else:
        await websocket.send("Message:Wrong Key Words")

def ReceivedByteData(data):
    saveJPG(data)

async def SendYoloPoints(data):
    global Hololens_Client
    if not Hololens_Client == None:
        await Hololens_Client.send(data)

def SendHintPoint(color, posx, posy):
    global Hololens_Client
    if not Hololens_Client == None:
        message = "HintPoints:{0},{1},{2}".format(color, posx, posy)
        Hololens_Client.send(message)

async def main():
    server = await websockets.serve(handle_client, "172.20.10.5", 3000)  # Replace with your desired host and port
    print("Server started, listening on {0}".format(server.sockets[0].getsockname()))
    flask_task = asyncio.to_thread(run_flask_app)
    await asyncio.gather(server.wait_closed(), flask_task)

def run_flask_app():
    app.run(host="0.0.0.0", port=5000)  # Change host and port as needed

def saveJPG(data):
    fileName = "public/test.jpg"
    data_label = data[0]
    data_value = data[1:]
    if int(data_label) == 0: 
        fileName = "public/virtual.jpg"
    else:
        fileName = "public/reality.jpg"
        
    with open(fileName, "wb") as file:
        file.write(data_value)

def YoloDetect(target):
    model = YOLO("best.pt")
    #model = YOLO("yolov8n.pt")
    predicted = model.predict(
        source="public/virtual.jpg",
        save=False,
        device="cpu",
        verbose=False
    )
    result = "Message:none"
    for pre in predicted: # 一張圖片 predict 結果為一個 pre
        dict_name = pre.names # 取得 model 中所有 names 的 dictionary
        list_box = pre.boxes # 取得所有的 boxes
        # print(pre.orig_shape) => y, x
        for i in range(len(list_box.cls)): # for each predicted object
            key = int(list_box.cls[i])  # get the key of the object
            obj_name = dict_name[key] # get the name of the object
            if obj_name == target:
                x_min = float(list_box.xyxy[i][0])/pre.orig_shape[1]*2-1
                x_max = float(list_box.xyxy[i][2])/pre.orig_shape[1]*2-1
                y_min = float(list_box.xyxy[i][1])/pre.orig_shape[0]*2-1
                y_max = float(list_box.xyxy[i][3])/pre.orig_shape[0]*2-1
                result = "YoloResult:{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f}".format(obj_name, x_min, y_min, x_max, y_max)
                break
    return result

if __name__ == "__main__":
    asyncio.run(main())