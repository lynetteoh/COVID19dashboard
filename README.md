# COVID19-dashboard

## Requirements 
1. [PostgreSQL 12.4](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)
2. [Python 3.8.5](https://www.python.org/downloads/release/python-385/)
3. [Cuda 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)


## Instructions to setup development environment.

**Make sure you have the above required software before setup development environment 


1. Update/install virtualenv using pip: pip install virtualenv
```
virtualenv venv -p python3.8
```

2. Run the development environment:
```
source venv/bin/activate
```

3. Install dependency:
```
pip install -r requirements.txt
```

## Instructions to create postgresql with docker 
1. Pull docker image 
   ```
   docker pull postgres
   ```
2. run docker container
   ```
    docker run --name covid19-postgres -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -d postgres
   ```

## Instructions for running the website locally:

Run the server and open localhost on a web browser:
```
python manage.py runserver
```


## Every time you add a python dependancy to the project, run:
```
pip freeze > requirements.txt
```

## Wiping database clean and load data into database

1. remove all files EXCEPT __init__.py files in migration folders for each app.
2. Drop the current database or drop tables in current database

> #### Drop database:
    ```
    psql postgres
    DROP DATABASE covid19;
    CREATE DATABASE covid19;
    ```

> #### Remove table in database: 

    To clean up data in postgres database

    ```
    psql covid19
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    select current_user;
    GRANT ALL ON SCHEMA public TO <current_user>;
    GRANT ALL ON SCHEMA public TO public;
    ```

3. Create the initial migrations and generate the database schema:

```
python manage.py makemigrations

python manage.py migrate
```

4. Load data to database 
``` 
python manage.py runscript load_data
```



