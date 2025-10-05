
# Car Type Classification - Assignment 3

This repository contains a web application for **Car Type Classification**.

---

## Getting Started

Follow these steps to pull the project into VS Code, run the web application using Docker, and test the code.

### 1. Clone the repository
Open your terminal in VS Code and run:

```bash
git clone https://github.com/supipivirajini96-maker/car-type-classification-assignment3.git
cd car-type-classification-assignment3
````

### 2. Start the application using Docker Compose

Run:

```bash
docker-compose up
```

This command will **build and start the Docker container** with the application.

### 3. Open the web application

Once the Docker container is running, open your browser and navigate to:

```
http://localhost:8050/
```

You should see the Car Type Classification web application running.


### 4. Run Unit Tests

To test the application, in the terminal run the following command:

```bash
python -m pytest code/test_app.py -v
```
