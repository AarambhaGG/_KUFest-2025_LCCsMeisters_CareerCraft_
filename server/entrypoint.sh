#!/bin/sh

# Exit on error
set -e

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput || true

echo "Starting server..."
exec python manage.py runserver 0.0.0.0:8000
