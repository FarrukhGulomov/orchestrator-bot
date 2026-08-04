# Used for BOTH local dev (docker-compose.yml) AND production on Railway —
# Railway auto-detects a Dockerfile at the repo root and builds from it
# INSTEAD of nixpacks whenever one is present (that's Railway's own
# builder-selection rule, not something set in this repo). An earlier
# version of this comment claimed nixpacks was always used and this file
# was dev-only; that was wrong from the moment this file was added and
# cost real debugging time chasing a NIXPACKS_APT_PKGS env var that was
# never actually in the build path — see meeting_attendee.py's docstring
# for the full story. Treat this file as the real production build.
FROM python:3.11-slim

WORKDIR /app

# Separate layer from the app source so a rebuild only re-installs
# dependencies when requirements.txt actually changes.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# meeting_attendee.py (/uchrashuv, gated behind MEETING_BOT_ENABLED so it's
# a no-op cost for anyone who doesn't set that) needs OS-level packages pip
# can't provide: ffmpeg/PulseAudio for audio capture, xvfb for a virtual
# display (Chromium launches headED by default — meeting_bot_headless — so
# platforms don't fingerprint it as a headless bot; that needs an X server
# in a container with no desktop session, see meeting_attendee._ensure_display),
# and Chromium's own shared libraries. These need apt+root at BUILD time —
# the container runs as a non-root user below, so this can't happen later.
# Chromium's own browser binary is NOT fetched here; it downloads itself
# lazily on the first /uchrashuv call (see meeting_attendee._ensure_chromium).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg pulseaudio xvfb \
    && playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Runs as a non-root user — no reason a Telegram-polling process needs root.
RUN useradd --create-home --uid 1000 bot && chown -R bot:bot /app
USER bot

CMD ["python", "bot.py"]
