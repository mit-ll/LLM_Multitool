#!/bin/bash
set -e
cd /S/llm_ui

# For Cystal LLM web deployment:
# bundle exec rake assets:precompile
# bundle exec rails server -b 0.0.0.0

cd /io
exec "$@"

