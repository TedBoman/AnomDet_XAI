# AnomDet - A Performance Anomaly Detector

## About The Project

Anomaly detection of real-world data consists of recognizing outlier data points. These outliers are what's called anomalies and there are plenty of algorithms with the purpose of finding these anomalies. The performance of these algorithms can be very dependent on the dataset it is used on. This means that if an algorithm performs well on one data set, it doesn't necessarily mean that it performs well on another.

AnomDet is a system which can manage different anomaly detection algorithms and anomaly injection methods by either simulating a real-time data stream or by reading data in batches. With AnomDet, you're provided a working framework for evaluating the performance of pre-defined anomaly detection models and how they respond to pre-defined anomaly injections. The framework architecture is modular by design and a simple way to add your own models and anomaly detection methods are provided in [Usage](#Usage) 

## Getting started

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
docker exec -it TSdatabase psql -U AnomDet -d mytimescaleDB
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
-Database mytimescaleDB
-Username AnomDet
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
    
## Usage

### Adding a model

### Adding a injection method

## License

This project is licensed under Creative Commons Attribution 4.0 International. See `LICENCE` for more details. 

## Authors

- [MarcusHammarstrom](https://github.com/MarcusHammarstrom)
- [Liamburberry](https://github.com/Liamburberry)
- [TedBoman](https://github.com/TedBoman)
- [MaxStrang](https://github.com/MaxStrang)
- [SlightlyRoasted](https://github.com/SlightlyRoasted)
- [valens-twiringiyimana](https://github.com/valens-twiringiyimana)
- [Seemihh](https://github.com/Seemihh)

## Acknowledgements