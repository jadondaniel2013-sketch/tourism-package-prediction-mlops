# Use Python 3.11 slim to match the Python version selected in GitHub Actions and Streamlit Community Cloud.
FROM python:3.11-slim
# Disable creation of .pyc files so the container remains cleaner and easier to inspect.
ENV PYTHONDONTWRITEBYTECODE=1
# Force Python output to be unbuffered so deployment logs appear immediately.
ENV PYTHONUNBUFFERED=1
# Set the container working directory for all following deployment commands.
WORKDIR /app
# Copy the deployment dependency file first so Docker can reuse its dependency layer when only app code changes.
COPY requirements.txt /app/requirements.txt
# Install the exact Python packages needed by the Streamlit deployment.
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the Streamlit app, model artifact, and smoke-test script into the image.
COPY . /app
# Document the port used by Streamlit inside the container.
EXPOSE 8501
# Launch the Streamlit application and listen on every interface inside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
