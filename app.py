#import necessity
from flask import Flask, render_template, jsonify
from main import Main
import threading
import logging
import csv

#create app
app=Flask(__name__)

#reset local logging and queue storage
with open("my_log.txt", "w") as f:
    pass
with open("queue.csv", "w") as f:
    pass

#init logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("my_log.txt")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)


#run the tetris frontend 
@app.route('/')
def tetris():
    return render_template('tetris.html')

#handle move queue requests from frontend
@app.route("/get_data")
def get_data():
    with open('queue.csv', "r") as f:
        reader = csv.reader(f)
        moveQueue =[]
        for row in reader:
            moveQueue.append(row)

    response = jsonify({"moveQueue": moveQueue})
    return response

#handle move queue updates from frontend
@app.route("/set_data", methods=["POST"])
def set_data():
    with open('queue.csv', "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    data.pop(0)
    with open("queue.csv", "w") as csvfile:
        pass
    return jsonify({"message": "Data set successfully."})

#keep the latest move in the queue uptodate
def pred():
    main = Main()
    gen = main.predict()
    while True:
        with open('queue.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([next(gen)])
            
#Run the app with a thread that handles the gesture recog in the background and another that handles the flask app.
if __name__ == "__main__":
    thread=threading.Thread(target=pred)
    thread.start()
    app.run(debug=True)