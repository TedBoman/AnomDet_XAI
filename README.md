# AnomDet - A Performance Anomaly Detector

## About The Project

## Getting started

    * Prerequisites
    - First of all, before you begin, ensure that you have Docker Desktop installed on your system.
    - Ensure that you have Git installed on your system.

    * Step 1: Clone the repository

        - Clone the repository using a terminal, run the following command:
        ``` 
        git clone https://github.com/MarcusHammarstrom/AnomDet.git
        ```
        Open the folder in your choice of environment and navigate into the project folder:
        ``` 
        >cd Docker
        ```
    * Step 2: Build and Start the Docker Container

        Run the following command to build and start the Docker Container:
        ```
        docker-compose up -d
        ```
        What this command does:
            ```
            -Downloads the required Docker images if they aren't already installed on your machine.
            -Starts a PostgreSQL database container.
            - The "-d" flag makes sure the container runs in the background.
            ```
    * Step 3: Access the PostgreSQL Database

        [Option 1]
        To access the database from within the Docker container, run the following command:
        ```
        docker exec -it timescaledb psql -U AnomDet -d mytimescaleDB
        ```
        What this command does and what the flags are for:
            ```
            -Opens an interactive psql session connected to the database.
            - "timescaledb" is the name of the running container.
            - "-U AnomDet" specifies the PostgreSQL user.
            - "-d mytimescaleDB" specifies the database name.
            ```
        To exit the psql session, just type:
            ```
            \q
            ```
        [Option 2]
        To access the database using an External PostgreSql Client such as "pgAdmin" or "DBeaver" use the following credentials:
            ```
            -Host localhost
            -Port 5432
            -Database mytimescaleDB
            -Username AnomDet
            -Password ******
            ```
    * Step 4: Stopping the Container
        
        To stop the container without removing its data, run the following command in the terminal:
            ```
            docker-compose down
            ```

        [Optional]
        If you need to access the running container's shell for debugging or inspecting, run the following command in the terminal:
            ```
            docker exec -it timescaledb bash
            ```
    
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