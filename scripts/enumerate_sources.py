"""Standalone probe: list every Media Frame Source Group and its sensor kinds.

    python scripts/enumerate_sources.py
"""
import asyncio
from winsdk.windows.media.capture.frames import MediaFrameSourceGroup


async def main():
    groups = await MediaFrameSourceGroup.find_all_async()
    print(f"Found {len(groups)} media frame source group(s)\n")
    for g in groups:
        print(f"GROUP: {g.display_name}")
        print(f"   id: {g.id}")
        for si in g.source_infos:
            print(f"   - kind={si.source_kind.name:9s} "
                  f"stream={si.media_stream_type.name}  id={si.id}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
