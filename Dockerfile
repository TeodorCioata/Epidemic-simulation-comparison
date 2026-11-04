#Dockerfile is essential for containerizing the application.
#Conteirization packages code and dependencies to ensure reproducibility across different machines.

#Specify the Python base image
FROM python:3.12-slim

#Configure Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app

#Set the working directory for the container
WORKDIR /app

#Copy the files from the backed that will be needed for the project
COPY pyproject.toml README.md ./
COPY episim/ ./episim/
COPY main.py countermeasures.py run_experiments.py ./
#Copy the premade figures
COPY docs/figures/ ./docs/figures/

#Install the project and the dependencies
RUN pip install --no-cache-dir .

#Copy the files from the frontend that will be needed for the project
COPY frontend/ ./frontend/

#This is the port for using Streamlit
EXPOSE 8501

#Commands to start the Streamlit application and access it outside the container
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
