FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 8080
CMD ["python", "-m", "dynamic_graph_fed_rl.saas.app"]
