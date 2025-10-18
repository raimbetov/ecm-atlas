FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV HOST=0.0.0.0
ENV DEBUG=False
ENV DATA_PATH=08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

EXPOSE 8080

CMD ["python", "10_unified_dashboard_2_tabs/api_server.py"]
