#!/bin/bash

set -e

# now fix perms on the mounted volume
chown -R appuser:appuser /app/assets
# drop into the normal CMD
exec gosu appuser "$@"
