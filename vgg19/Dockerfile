# Use the official Python image
FROM python:3.8

# Set the working directory
WORKDIR /vgg19

# Copy the Python script and the required files for both services
COPY vgg19.py .
COPY requirements-vgg19.txt .
# Copier le dossier Data dans le conteneur



# Install dependencies
RUN pip install -r requirements-vgg19.txt


# Expose the service ports
EXPOSE 800

# Start the appropriate service based on an environment variable
CMD ["python", "vgg19.py"]
