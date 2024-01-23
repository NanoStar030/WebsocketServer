import asyncio
import websockets
from ultralytics import YOLO

Hololens_Client = None
models = {"PCB Residue": "pcb.pt", "gas-valve": "gas.pt", "on-off": "onoff.pt", "EMS":"best.pt", "LASER": "best.pt"}
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

async def main():#192.168.1.171
    server = await websockets.serve(handle_client, "172.20.10.7", 3000)  # Replace with your desired host and port 192.168.50.9
    print("Server started, listening on {0}".format(server.sockets[0].getsockname()))
    await server.wait_closed()

def saveJPG(data):
    fileName = "public/reality.jpg"
    with open(fileName, "wb") as file:
        file.write(data)

def YoloDetect(target):
    model = YOLO(models[target])
    #model = YOLO("yolov8n.pt")
    predicted = model.predict(
        source="public/reality.jpg",
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
            if target == "LASER":
                if obj_name == "LASER":
                    x_min = float(list_box.xyxy[i][0])/pre.orig_shape[1]*2-1
                    x_max = float(list_box.xyxy[i][2])/pre.orig_shape[1]*2-1
                    y_min = float(list_box.xyxy[i][1])/pre.orig_shape[0]*2-1
                    y_max = float(list_box.xyxy[i][3])/pre.orig_shape[0]*2-1
                    result = "YoloResult:{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f}".format(obj_name, x_min, y_min, x_max, y_max)
                    print(result)
                    break
            elif target == "EMS":
                if obj_name == "EMS":
                    x_min = float(list_box.xyxy[i][0])/pre.orig_shape[1]*2-1
                    x_max = float(list_box.xyxy[i][2])/pre.orig_shape[1]*2-1
                    y_min = float(list_box.xyxy[i][1])/pre.orig_shape[0]*2-1
                    y_max = float(list_box.xyxy[i][3])/pre.orig_shape[0]*2-1
                    result = "YoloResult:{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f}".format(obj_name, x_min, y_min, x_max, y_max)
                    print(result)
                    break
            else:
                x_min = float(list_box.xyxy[i][0])/pre.orig_shape[1]*2-1
                x_max = float(list_box.xyxy[i][2])/pre.orig_shape[1]*2-1
                y_min = float(list_box.xyxy[i][1])/pre.orig_shape[0]*2-1
                y_max = float(list_box.xyxy[i][3])/pre.orig_shape[0]*2-1
                result = "YoloResult:{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f}".format(obj_name, x_min, y_min, x_max, y_max)
                print(result)
                break
    return result

if __name__ == "__main__":
    asyncio.run(main())