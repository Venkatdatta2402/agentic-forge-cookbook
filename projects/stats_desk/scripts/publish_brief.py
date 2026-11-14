"""Show a brief to everyone as an example, or stop showing it (the store chosen by STATS_DESK_STORE;
Firestore when deployed).

    uv run python scripts/publish_brief.py <brief id>          # public
    uv run python scripts/publish_brief.py <brief id> --off    # by its link only again
"""

import asyncio
import sys

from stats_desk.api.store import from_env


async def main(brief_id, public):
    if not await from_env().set_public(brief_id, public):
        sys.exit(f"No brief {brief_id}.")
    print(f"{brief_id}: {'public example' if public else 'by its link only'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--off"]
    if len(args) != 1:
        sys.exit(__doc__)
    asyncio.run(main(args[0], "--off" not in sys.argv))
