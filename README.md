# AnomDet - A Performance Anomaly Detector

## 📑 Table of contents

- [About The Project](#-about-the-project)
- [How To Build](#-how-to-build)
- [Tools And Frameworks](#-tools-and-frameworks)
- [Guide](#-guide)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgements](#-acknowledgements)

## 💻 About The Project

### Overview

Anomaly detection of real-world data consists of recognizing outlier data points. These outliers are what's called anomalies and anomaly detection algortihms have been researched extensively. The performance of these algorithms can be very dependent on the dataset it is used on. This means that if an algorithm performs well on one data set, it doesn't necessarily mean that it performs well on another.

AnomDet is a system which can manage different anomaly detection algorithms and anomaly injection methods by either simulating a real-time data stream or by reading data in batches. With AnomDet, you're provided a working framework for evaluating the performance of pre-defined anomaly detection models and how they respond to pre-defined anomaly injections. 

The system architecture is modular by design and a simple way to add your own models and anomaly detection methods are provided in our [Guide](#-guide). If the choice of database doesn't suit you, we have abstracted away the API between the backend and the database with an interface described in our [Guide](#-guide) under [Database APi](#-database-api).

### Features provided

AnomDet allows for anomaly detection by importing a complete dataset in one batch or by simulating a real-time stream of the imported data. A machine learning model will process the data that is fed to the system and visualize the results in a frontend.

The system also provides a set of self-defined anomaly detection algorithms and anomaly injection methods. The backend API offers the user to initiate a batch or stream job from the frontend or by invoking it from the command line. The API provides ways to check what models and injection methods are provided as well as listing jobs currently running. More details can be found in the [Guide](#-guide).

## 📝 How To Build

* Prerequisites
- First of all, before you begin, ensure that you have Docker Desktop installed on your system.
- Ensure that you have Git installed on your system.

* Step 1: Clone the repository

- Clone the repository using a terminal, run the following command:
```sh 
git clone https://github.com/MarcusHammarstrom/AnomDet.git
```
Open the folder in your choice of environment and navigate into the project folder:
```sh 
cd Docker
```
* Step 2: Build and Start the Docker Container

Run the following command to build and start the Docker Container:
```sh
docker-compose up -d
```
What this command does:
```sh
-Downloads the required Docker images if they aren't already installed on your machine.
-Starts a PostgreSQL database container.
- The "-d" flag makes sure the container runs in the background.
```
* Step 3: Access the PostgreSQL Database

[Option 1]
To access the database from within the Docker container, run the following command:
```sh
docker exec -it TSdatabase psql -U Anomdet -d TSdatabase
```
What this command does and what the flags are for:
```sh
-Opens an interactive psql session connected to the database.
- "timescaledb" is the name of the running container.
- "-U AnomDet" specifies the PostgreSQL user.
- "-d mytimescaleDB" specifies the database name.
```
To exit the psql session, just type:
```sh
\q
```
[Option 2]
To access the database using an External PostgreSql Client such as "pgAdmin" or "DBeaver" use the following credentials:
```sh
-Host localhost
-Port 5432
-Database TSdatabase
-Username Anomdet
-Password ******
```
* Step 4: Stopping the Container
        
To stop the container without removing its data, run the following command in the terminal:
```sh
docker-compose down
```

[Optional]
If you need to access the running container's shell for debugging or inspecting, run the following command in the terminal:
```sh
docker exec -it timescaledb bash
```

## 🛠️ Tools And Frameworks

### Python

### Docker

### TimescaleDB

### Dash

## 📚 Guide

### Frontend

### CLI-tool

### Adding a model

### Adding an injection method

### Database API

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

## 👏 Acknowledgements