# AnomDet - A Performance Anomaly Detector

## 📑 Table of contents

- [About The Project](#-about-the-project)
- [How To Build](#-how-to-build)
- [Tools And Frameworks](#%EF%B8%8F-tools-and-frameworks)
- [Guide](#-guide)
- [For Developers](#-for-developers)
- [License](#-license)
- [Authors](#-authors)

## 💻 About The Project

### Overview of AnomDet_XAI
This repo is a fork of the AnomDet with the visualization library switched to Grafana and integration of XAI to explain anomalies.

### Overview of AnomDet

Anomaly detection of real-world data consists of recognizing outlier data points. These outliers are what's called anomalies and anomaly detection algortihms have been researched extensively. The performance of these algorithms can be very dependent on the dataset it is used on. This means that if an algorithm performs well on one data set, it doesn't necessarily mean that it performs well on another.

AnomDet is a system which can manage different anomaly detection algorithms and anomaly injection methods by either simulating a real-time data stream or by reading and processing data in one batch. With AnomDet, you're provided a working framework for evaluating the performance of pre-defined anomaly detection models and how they respond to pre-defined anomaly injections. The system is also designed in such a way that a user can easily define and integrate their own detection models and injection methods.

How to interact with our system through our [Frontend](#frontend) and [CLI-tool](#cli-tool) is covered under [Guide](#-guide).

Since the system architecture is modular, we have a [For Developers](#-for-developers) section that covers how to change frontend, adding detection models, adding injection methods, changing database manager and working with our API's.

### Features provided

AnomDet allows for anomaly detection by importing a complete dataset in one batch or by simulating a real-time stream of the imported data. A machine learning model will process the data that is fed to the system and label the data normal or anomalous. The results can then be visualized in a frontend.

The system also provides a set of self-defined anomaly detection algorithms and anomaly injection methods. By instatiating a backend API object our Frontend and CLI-tool offers users two ways of interacting with the system. The API provides ways to check what models and injection methods are provided as well as listing jobs currently running. More details can be found in the [Guide](#guide).

### Westermo test system performance data set

To develop our product, to evaluate our models and to test our system during development, we have used Westermo test system performance data set retrieved from: [https://github.com/westermo/test-system-performance-dataset](https://github.com/westermo/test-system-performance-dataset). This dataset contains anonymized data from 19 of [Westermo's](https://www.westermo.com/) test servers with more than 20 performance metrics such as CPU and memory usage. 

## 📝 How To Build

### Installation

1. Install and run Docker Desktop or install Headless Docker on the system of your choice
2. Ensure that you have Git installed on your system
3. Clone the repository using a terminal running the following command:
   ```sh 
   git clone https://github.com/MarcusHammarstrom/AnomDet.git
   ```
4. Navigate to the cloned repository and change to the Docker directory
   ```sh 
   cd Docker
   ```
5. Create a .env file in the Docker directory
6. Set up the following environment variables:
   ```
   DATABASE_USER=<your-db-user>
   DATABASE_PASSWORD=<your-db-password>
   DATABASE_NAME=<your-db-name>
   DATABASE_HOST=<your-db-hostname>
   DATABASE_PORT=<your-db-port>
   FRONTEND_PORT=<your-frontend-port>
   BACKEND_PORT=<your-backend-port>
   BACKEND_HOST=<your-backend-hostname>
   ```
7. Run the following command to build and start the Docker container:
   ```sh
   docker-compose up -d
   ```
8. Your system should now be built and the system is ready 

   ![Terminal output from build](./images/terminal_output.png)

### Additional Commands

- To access the database from within the Docker container, run the following command:
  ```sh
  docker exec -it TSdatabase psql -U <your-db-user> -d TSdatabase
  ```
- To exit the psql session, just type:
  ```sh
  \q
  ```
- To stop the containers without removing its data, run the following commands in the terminal:
  ```sh
  docker-compose down
  ```
- To stop the containers and remove its data, run the following commands in the terminal:
  ```sh
  docker-compose down -v
  ```

- If you need to access the running container's shell for debugging or inspecting, run the following command in the terminal:
  ```sh
  docker exec -it timescaledb bash
  ```

## 🛠️ Tools And Frameworks

### Python

The entire stack of our system is developed using the python programming language and python is the de facto standard when it comes to machine learning. The initial stakeholder of the system also told us that it would be nice to have the whole system developed using python which made this choice easy.

### Docker

As per the request from the initial stakeholder, the system should have a nice architecture with containerized components. Docker is a popular and well documented tool to run software in containers.

### Dash

As we wanted a frontend as a webUI to interact with the system and visualize data, developed in python, we decided to use Dash. Dash is a python web framework made by Plotly to create interactive web applications to visualize data.

### TimescaleDB

As a functional requirement by the initial stakeholder, they wanted us to store time-series data in a time-series database. Our choice was TimescaleDB which is an open-source time-series database that extends PostgreSQL by adding hypertables optimized for time-series data.

### Libraries

- Pandas - For file reading and manipulation
- Bokeh   - For plotting our data in our webUI
- Tensorflow Keras - A powerful machine learning library used to create neural networks
- scikit learn - Machine learning library providing models ready for training
- psycopg2 - A PostgreSQL database adapter for python


## 📚 Guide

Users can interact with our system in two ways, through the frontend interface or the CLI-tool.

### Frontend

The Frontend is split up into two different pages. On the main page, the user can start a batch or stream job or view a list of already started jobs. To start a job, the user has different options such as dataset to use, model to use and whether or not to use injection. After filling out the necessary input fields, the user can press the "Start Job" button to start a job. 

At the bottom of the page there is a list of currently running jobs. This list displays the name of each running job and the text label of the job is a link to go view that job. By pressing the link, you get redirected to a page that displays the data specific to that job. Next to the link there is a stop button, this button will stop a running job. 

The data visualization page's main focus is to visualize the data in each column of the dataset and display if it’s an anomaly or not. The data visualization page displays the first three columns by default, unless the dataset has less than three columns, then all columns are displayed. The user then has a list of checkboxes to check or uncheck to choose to display or remove a plot. Finally you have a “Back to Home” button to navigate back to the main page. 

### CLI-tool

A installed and running system can be interacted with through the command line by navigating to the backend API folder and invoking "cli_tool.py":
```sh
cd Backend/api
python cli_tool.py help
```
The output from running the help command describes all functionality provided by the CLI-tool to interact with the system:
![CLI-tool help output](images/cli_help_output.png)
Running a "run-batch" or "run-stream" will give you further inputs to define before a requests is sent to the backend:
![Starting batch job with CLI-tool](images/cli_run_batch.png)

## ☕ For Developers

### Adding a model

AnomDet offers a simple way to integrate new detection models. To integrate a new model you have to define a new model class that inherits from this abstract class:
```py
class ModelInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, df):
         pass

    @abstractmethod
    def detect(self, detection_df):
         pass
```
The abstract class is not strict at all, it defines a constructor where you should instantiate a model object with parameters of your choosing. In the "run" method, your model should do all the necessary preprocessing of the data and train your model. The "detect" method should label each row of data in "detection_df" and return it as a list of predictions.

Finally, you need to modify the "get_model" script to provide your model as an option to the system. You do this by importing your model class as well as modifying the get_model function:
```py
def get_model(model):
    match model:
        case "lstm":
            lstm_instance = lstm.LSTMModel()
            return lstm_instance
            
        case "isolation_forest":
            if_instance = isolation_forest.IsolationForestModel()
            return if_instance
            
        case "svm":
            svm_instance = svm.SVMModel()
            return svm_instance
```

### Adding an injection method

The anomaly injector has been made modular to allow for easy addition of anomaly injection methods. To add a method, create a new python file in the `/Backend/Simulator/AnomalyInjector/InjectionMethods` directory. Here you will need to use the following template:

```py
class YourAnomalyInjectionMethod():

   def inject_anomaly(self, data, rng=None, data_range=None, mean=None, magnitude=None):
      your_logic
      return data
```

You are free to use any imports and as many functions as you need to produce your anomaly. As the anomaly injector works with pandas dataseries (dataframe in the injectionmethods) your code must be able to work with these aswell. It is not mandatory to create classes for the methods however, this makes it easier to allow for more sophisticated methods.

- **data: Pandas DataFrame/Series**. data is the pandas dataframe containing the data. It looks like this [row-number, column-data]. So you will just need to perform calculations on the dataframe as if it was a normal variable.
- **rng: NumPy random number generator (optional)**: is a pre-seeded random class that will allow for the methods to have some randomness to them.
- **mean: float (optional)**: is the mean between all the data in the column that the anomaly will be injected on. 
- **data_range (optional): float**: is the value range from the min value to the max value. If the min value is 5 and the max is 12, the data_range will be 7.
- **magnitude: float (optional)**: is the user-specified scalar value used to scale the anomaly value.

The next step in integrating your injection method is to import your class. In the `/Backend/Simulator/AnomalyInjector/anomaly_injector.py` file, add your class to the top list of file. 

```py
from Simulator.AnomalyInjector.InjectionMethods.your_method_name import YourAnomalyInjectionMethod
```

Now the class is ready to be used in the anomaly injector. Add your method to the _apply_anomaly method. In this method, we have a list of if and elif statements. Simply add your own elif statement here:

```py
elif anomaly_type == 'your_method_name':
    injector = YourAnomalyInjectionMethod()
    result = injector.inject_anomaly(data, magnitude)
    dl.debug_print(f"New data after custom anomaly: {result}")
    return result
```

Currently, the anomaly injector will give the method access to one row of data to reduce data dependency All the code is wrapped in try-except statements meaning that is you start a job with debugging enabled or disabled in the cli-tool, any errors will be printed to the backend terminal. 

For easier confirmation if your anomaly works and if it produces the correct values, you can enable the debug parameter in the cli-tool when starting a job. This will print out all the values of the data before and after injecting an anomaly on it. 

### Backend API

Since our system provides information to the Frontend through a generalized API, it is easy to create your own Frontend to interact with the system rather than the one provided. All necessary information provided to the Frontend is accessed by sending requests to the backend and no system information is stored in the Frontend.

The Backend API methods are:
```py
def __init__(self, host: str, port: int) -> None:
def run_batch(self, model: str, dataset: str, name: str, inj_params: dict=None) -> None:
def run_stream(self, model: str,dataset: str, name: str, speedup: int, inj_params: dict=None) -> None:
def get_data(self, timestamp: str, name: str) -> str:
def get_running(self) -> str:
def cancel_job(self, name: str) -> None:
def get_models(self) -> str:
def get_injection_methods(self) -> str:
def get_datasets(self) -> str:
def get_all_jobs(self) -> str:
def import_dataset(self, file_path: str, timestamp_column: str) -> None:
```

More detailed documentation on the API itself can be found in our [API_README](https://github.com/MarcusHammarstrom/AnomDet/blob/main/Backend/API_README.md)

To add your own Frontend to the system, run your frontend in a container or your hosting method of choice and then interact with the Backend with this API.

### Database API

Since we have designed a database interface for our system to be more modular, changing database manager does not affect the rest of the system. To change database manager, all that is needed is to provide an API that follows our database interface and then provide the right connection parameters when instantiating a API object. 

The Database API interface has the following methods defined:
```py
class DBInterface(ABC):
    # Constructor that adds all connection parameters needed to connect to the database to the object
    # conn_params is a dictionary with the parameters needed for the specific database implementation
    # As an example structure, the dictionary for a TimescaleDB implementation could look like this:
    # conn_params = {
    #     "user": "username",
    #     "password": "password",
    #     "host": "hostname",
    #     "port": "port",
    #     "database": "database"
    # }
    @abstractmethod
    def __init__(self, conn_params: dict):
        pass
    # Creates a hypertable called table_name with column-names columns
    # First column of name columns[0] is of type TIMESTAMPTZ NOT NULL and the rest are VARCHAR(50)
    # Then two new columns of type BOOLEAN are added to the table, is_anomaly and injected_anomaly
    @abstractmethod
    def create_table(self, table_name: str, columns: list[str]) -> None:
        pass
    # Inserts data into the table_name table. The data is a pandas DataFrame with matching columns to the table
    @abstractmethod
    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        pass
    # Reads each row of data in the table table_name that has a timestamp greater than or equal to time
    @abstractmethod
    def read_data(self, time: datetime, table_name: str) -> pd.DataFrame:
        pass
    # Deletes the table_name table along with all its data
    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        pass
    # Checks if the table_name table exists in the database
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        pass
    # Returns a list of all columns in the table_name table
    @abstractmethod
    def get_columns(self, table_name: str) -> list[str]:
        pass
    # Updates rows of the table that have an anomaly detected
    @abstractmethod
    def update_anomalies(self, table_name: str, anomalies: pd.DataFrame) -> None:
        pass
```

To use your own database manager of choice, run it in a container or in your preferred environment. Then implement a Database API that follows the interface specified. Provide the right credentials to the backend and instantiate  an object of your Database API class in the backend.

## Copyright
### OmniXAI
Copyright (c) 2021, Salesforce.com, Inc.
All rights reserved.

## 📄 License

This project is licensed under Creative Commons Attribution 4.0 International. See `LICENCE` for more details. 

## ✍ Authors

- [MarcusHammarstrom](https://github.com/MarcusHammarstrom)
- [Liamburberry](https://github.com/Liamburberry)
- [TedBoman](https://github.com/TedBoman)
- [MaxStrang](https://github.com/MaxStrang)
- [SlightlyRoasted](https://github.com/SlightlyRoasted)
- [valens-twiringiyimana](https://github.com/valens-twiringiyimana)
- [Seemihh](https://github.com/Seemihh)
