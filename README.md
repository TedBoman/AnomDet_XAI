# AnomDet - A Performance Anomaly Detector

## 📑 Table of contents

- [About The Project](#-about-the-project)
- [How To Build](#-how-to-build)
- [Tools And Frameworks](#-tools-and-frameworks)
- [Guide](#-guide)
- [For Developers](#-for-developers)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgements](#-acknowledgements)

## 💻 About The Project

### Overview

Anomaly detection of real-world data consists of recognizing outlier data points. These outliers are what's called anomalies and anomaly detection algortihms have been researched extensively. The performance of these algorithms can be very dependent on the dataset it is used on. This means that if an algorithm performs well on one data set, it doesn't necessarily mean that it performs well on another.

AnomDet is a system which can manage different anomaly detection algorithms and anomaly injection methods by either simulating a real-time data stream or by reading and processing data in one batch. With AnomDet, you're provided a working framework for evaluating the performance of pre-defined anomaly detection models and how they respond to pre-defined anomaly injections. The system is also designed in such a way that a user can easily define and integrate their own detection models and injection methods.

How to interact with our system through our [Frontend](#frontend) and [CLI-tool](#cli-tool) is covered under [Guide](#-guide). Also covered in our [Guide](#-guide) is how the interaction with the [Backend API](#backend-api) and the [Database API](#database-api) works.

Since the system architecture is modular, we have a [For Developers](#-for-developers) section that covers how to change frontend, adding detection models, adding injection methods and changing database manager.

### Features provided

AnomDet allows for anomaly detection by importing a complete dataset in one batch or by simulating a real-time stream of the imported data. A machine learning model will process the data that is fed to the system and label the data normal or anomalous. The results can then be visualized in a frontend.

The system also provides a set of self-defined anomaly detection algorithms and anomaly injection methods. By instatiating a backend API object our Frontend and CLI-tool offers users two ways of interacting with the system. The API provides ways to check what models and injection methods are provided as well as listing jobs currently running. More details can be found in the [Guide](#guide).

## 📝 How To Build

### Installation

1. Install Docker Desktop
2. Ensure that you have Git installed on your system
3. Clone the repository using a terminal running the following command:
   ```sh 
   git clone https://github.com/MarcusHammarstrom/AnomDet.git
   ```
4. Navigate to the cloned repository and change to the Docker directory
   ```sh 
   cd Docker
   ```
5. Create .env file in the Docker directory
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
![Terminal output from build](image.png)

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

### Frontend

### CLI-tool

### Backend API

### Database API

## ☕ For Developers

### Adding a model

### Adding an injection method

### Changing Frontend

Since our system provides information to the Frontend through a generalized API, it is easy to create your own Frontend to interact with the system rather than the one provided. All necessary information provided to the Frontend is accessed by sending requests to the backend and no system information is stored in the Frontend.

### Migrating to a different database manager

Since we have designed a database interface for our system to be more modular, changing database manager does not affect the rest of the system. To change database manager, all that is needed is to provide an API that follows our database interface and then provide the right connection parameters when instantiating a API object.

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
