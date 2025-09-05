# Starting with python base image
FROM python:3.9-slim

# Setting the working directory inside the container
WORKDIR /app

# Let's copy the requirements.txt into the container
COPY ./backend/requirements.txt /app/requirements.txt

# Let's install the dependencies (Helps keep the size smaller)
RUN pip install --no-cache-dir -r requirements.txt

# Let's copy the entire project code into the container
COPY . /app/

# Setting the port for jupyterlab
EXPOSE 8888

# Default command to run when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]