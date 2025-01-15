from get_handler import get_handler
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from time import sleep
import os 
import json
import shutil
import threading
import pandas as pd

active_jobs = []
update_lock = threading.Lock()

def main():
    global active_jobs
    print("Starting graph updater...")
    handler = get_handler()
    updating_thread = threading.Thread(target=update_stream_graphs, args=())
    updating_thread.daemon = True
    updating_thread.start()

    while handler.handle_get_running() is None:
        sleep(1)
    
    while True:
        current_jobs = handler.handle_get_running()
        current_jobs = json.loads(current_jobs)["running"]

        if len(current_jobs) > 0:
            print("Current jobs: ", current_jobs)
            with update_lock:
                for job in current_jobs:
                    if job not in active_jobs:
                        print("Creating graphs for job: ", job["name"])
                        create_graphs(job["name"], handler)
                        active_jobs.append(job)

            with update_lock:
                for job in active_jobs:
                    if job not in current_jobs:
                        delete_graphs(job["name"])
                        active_jobs.remove(job)
        
        sleep(3)

def create_graphs(job_name, handler):
    df = handler.handle_get_data(0, job_name)
    columns = df.columns.tolist()
    columns.remove("timestamp")
    columns.remove("is_anomaly")
    columns.remove("injected_anomaly")

    html_files = {}

    points_per_frame = 500
    print(df)
    
    if len(df) == 1:
        x_min = -1
        x_max = 1
    elif len(df) <= points_per_frame:
        x_min = df["timestamp"].min() * 0.9
        x_max = df["timestamp"].max() * 1.1
    else:
        max_time = df["timestamp"].max()
        while len(df[df["timestamp"] < max_time]) > points_per_frame:
            max_time *= 0.9
        x_min = df["timestamp"].min() * 0.9
        x_max = max_time
    x_range = (x_min, x_max)

    for col in columns:
        y = df[col]
        y_min = df[col].astype("float32").min()
        y_max = df[col].astype("float32").max()
        y_range = (y_min*0.8, y_max*1.2)

        true_normal = df[(df["is_anomaly"] == False) & (df["injected_anomaly"] == False)][["timestamp", col]]
        false_normal = df[(df["is_anomaly"] == False) & (df["injected_anomaly"] == True)][["timestamp", col]]
        anomalies = df[df["is_anomaly"] == True][["timestamp", col]]

        p = figure(
            width=1400, 
            height=350, 
            title=f"{col} timeline", 
            x_axis_label="Time", 
            y_axis_label=col, 
            x_range=(x_min, x_max),
            y_range=y_range,
            tools="pan,reset,save",
        )

        if len(true_normal) > 0:
            p.scatter(true_normal["timestamp"], true_normal[col], size=6, color="green", alpha=0.7, legend_label="Normal Data")
        if len(false_normal) > 0:
            p.scatter(false_normal["timestamp"], false_normal[col], size=6, color="blue", alpha=0.7, legend_label="Injected Anomalies Labeled as Normal", marker="diamond")  
        if len(anomalies) > 0:
            p.scatter(anomalies["timestamp"], anomalies[col], size=6, color="red", alpha=0.7, legend_label="Anomalies", marker="x")
    
        p.legend.location = "top_right"

        if x_max != df["timestamp"].max():
            p.x_range.bounds = (x_min, df["timestamp"].max() * 1.1)
        else:    
            p.x_range.bounds = x_range
        p.y_range.bounds = "auto"

        html_content = file_html(p, CDN, f"{col} Plot")
        html_files[col] = html_content

    directory = f"./graphs/{job_name}"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    for col in columns:
        print("Creating graph for column: ", col)
        file_handle = open(f"{directory}/{col}.html", "w")
        file_handle.write(html_files[col])
        file_handle.close()

def delete_graphs(job_name):
    directory = f"./graphs/{job_name}"
    if os.path.exists(directory):
        shutil.rmtree(directory)

def update_stream_graphs():
    handler = get_handler()
    while True:
        with update_lock:
            for job in active_jobs:
                if job["type"] == "stream":
                    create_graphs(job["name"], handler)
        sleep(10)

if __name__ == "__main__":
    main()