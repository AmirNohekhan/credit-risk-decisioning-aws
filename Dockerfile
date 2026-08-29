FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY scripts ./scripts
RUN mkdir -p artifacts
EXPOSE 8000
CMD ["uvicorn","credit_risk.serving:app","--host","0.0.0.0","--port","8000"]

