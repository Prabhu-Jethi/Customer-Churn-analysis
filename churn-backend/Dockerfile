## BASE IMAGE
FROM python:3.12-slim

## WORKING DIRECTORY
WORKDIR /app

## COPY
COPY requirements.txt .

## INSTALL REQUIREMENTS
RUN pip install --no-cache-dir -r requirements.txt

## COPY ALL FILES
COPY . .

## PORT 
EXPOSE 8000

## COMMANDS
CMD [ "sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}" ]
