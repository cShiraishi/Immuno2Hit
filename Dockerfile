# API container for Immuno2Hit. Vercel serves only the static frontend at the repo root;
# everything Python lives in backend/ so Vercel's runtime detection never fires on it.
FROM python:3.12-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY static/ ./static/

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
