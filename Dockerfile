FROM python:3.11

EXPOSE 8501

WORKDIR /app

COPY . .
COPY data/ /app/data/
COPY img/ /app/img/
COPY models/ /app/models/


RUN python -m pip install --upgrade pip && pip install -r requirements.txt

LABEL authors="enjoy@data-silence.com"
LABEL app_name='banks-clients'

CMD streamlit run app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS false \
    --browser.gatherUsageStats false