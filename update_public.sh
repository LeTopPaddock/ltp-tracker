#!/bin/bash
# Run this whenever you want to push updated picks to the public website.
# It copies the latest database and pushes to GitHub automatically.

set -e
cd "$(dirname "$0")"

echo "Copying latest database..."
cp ../sports-investing-bot/data/ltp_picks.db data/ltp_picks.db

echo "Pushing to GitHub..."
git add data/ltp_picks.db
git commit -m "Update picks database $(date '+%d %b %Y')"
git push

echo "Done — public site will update in ~30 seconds."
