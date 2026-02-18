import asyncio
import time
from animal import Animal, Config


async def run_smoke_test():
    config = Config(
        enable_anchor=False,
        debug=True,
        # log_verbosity=1  # uncomment for httpx summaries
        # log_verbosity=2  # uncomment for + fsspec, filelock
        # log_verbosity=3  # uncomment for full httpcore firehose
    )
    config.apply_logging()  # must be first — owns the logging setup

    engine = Animal(config)
    updates = []

    def subscriber(vibe):
        updates.append((time.time(), vibe))

    engine.subscribe(subscriber)

    # ------------------------------------------------------------------ startup
    await engine.start()
    await asyncio.sleep(0.5)

    # ------------------------------------------------------------------ process
    result = await engine.process_text(
        "I am absolutely thrilled to be here. This is incredible!"
    )

    assert "animal" in result, "Response missing 'animal' key"
    assert len(result["animal"]) == 5, (
        f"Expected 5 animal values, got {len(result['animal'])}"
    )

    # ------------------------------------------------------------------ decay
    await asyncio.sleep(1.5)

    # ------------------------------------------------------------------ stop
    await engine.stop()

    # ------------------------------------------------------------------ assert
    assert len(updates) > 5, (
        f"Engine did not tick properly — only {len(updates)} updates received"
    )
    assert any(v[1][0] > 0.5 for v in updates), (
        "Valence never exceeded 0.5 — emotional response may not be working"
    )

    print("✓ SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
