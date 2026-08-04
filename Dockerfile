# Local development image — see docker-compose.yml. Production on Railway
# builds from source with its own buildpack (nixpacks) and is NOT affected
# by this file; this exists purely so `docker compose up` gives a
# reproducible dev environment against real Postgres/Redis instead of the
# in-memory fallback tier every module degrades to without them.
FROM python:3.11-slim

WORKDIR /app

# Separate layer from the app source so `docker compose build` only
# re-installs dependencies when requirements.txt actually changes.
COPY requirements.txt requirements-dev.txt requirements-meeting.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Optional: meeting_attendee.py (MEETING_BOT_ENABLED=true) — joins a live
# Meet/Zoom/Teams call and records it. Off by default; installing this
# unconditionally would bloat every image with a Chromium download nobody
# else needs. Uncomment to build an image with it enabled:
#
# RUN pip install --no-cache-dir -r requirements-meeting.txt \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends ffmpeg pulseaudio xvfb \
#     && playwright install --with-deps chromium \
#     && rm -rf /var/lib/apt/lists/*

COPY . .

# Runs as a non-root user — no reason a Telegram-polling process needs root.
RUN useradd --create-home --uid 1000 bot && chown -R bot:bot /app
USER bot

CMD ["python", "bot.py"]
