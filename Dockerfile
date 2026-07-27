# API container for Immuno2Hit. Vercel's serverless functions cap at 250 MB and the
# dependencies alone are ~295 MB, so the Python side runs here (Hugging Face Spaces,
# Render, Fly.io) while Vercel serves the static frontend.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py predict.py fingerprints.py qsar_ad.py ./
COPY models/ ./models/
COPY static/ ./static/

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
