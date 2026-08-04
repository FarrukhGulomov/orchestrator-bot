# Local development image — see docker-compose.yml. Production on Railway
# builds from source with its own buildpack (nixpacks) and is NOT affected
# by this file; this exists purely so `docker compose up` gives a
# reproducible dev environment against real Postgres/Redis instead of the
# in-memory fallback tier every module degrades to without them.
FROM python:3.11-slim

WORKDIR /app

# Separate layer from the app source so `docker compose build` only
# re-installs dependencies when requirements.txt actually changes.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Optional: meeting_attendee.py (MEETING_BOT_ENABLED=true) — joins a live
# Meet/Zoom/Teams call and records it. The `playwright` pip package is
# already installed above (it's in requirements.txt, and it's small); what's
# still missing here is the OS side — ffmpeg/PulseAudio for audio capture
# and Chromium's shared libraries, which need apt+root at BUILD time (the
# container runs as a non-root user below, so this can't happen later).
# Chromium's own browser binary downloads itself lazily on first /uchrashuv
# call (see meeting_attendee._ensure_chromium) so it's not fetched here.
# Uncomment to build an image with the feature actually usable:
#
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends ffmpeg pulseaudio \
#     && playwright install-deps chromium \
#     && rm -rf /var/lib/apt/lists/*

COPY . .

# Runs as a non-root user — no reason a Telegram-polling process needs root.
RUN useradd --create-home --uid 1000 bot && chown -R bot:bot /app
USER bot

CMD ["python", "bot.py"]
