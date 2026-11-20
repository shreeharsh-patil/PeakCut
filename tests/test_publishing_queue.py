"""Tests for Multi-Platform Publishing Queue & Smart Scheduler (publishing_queue.py)."""
import datetime as dt
import time
from pathlib import Path

from engine.services.publishing_queue import PostItem, PublishingQueue


def test_queue_enqueue_and_persistence(tmp_path: Path):
    queue_file = tmp_path / "test_queue.json"
    queue = PublishingQueue(queue_file=queue_file, post_gap_minutes=30, daily_cap=3)

    item = queue.enqueue(
        job_id="job_1",
        clip_id="clip_0",
        platform="youtube",
        account_id="acc_yt_1",
        title="Epic Tech Short",
        video_path="output/clip_0.mp4",
        description="Check this out! #tech",
        schedule_now=True,
    )

    assert item.status == "scheduled"
    assert item.platform == "youtube"
    assert item.account_id == "acc_yt_1"
    assert item.due_at <= time.time()

    # Verify duplicate enqueue returns existing item (idempotency)
    dup = queue.enqueue(
        job_id="job_1",
        clip_id="clip_0",
        platform="youtube",
        account_id="acc_yt_1",
        title="Duplicate",
        video_path="output/clip_0.mp4",
    )
    assert dup.id == item.id
    assert len(queue.list_posts()) == 1

    # Reload from disk
    queue2 = PublishingQueue(queue_file=queue_file)
    assert len(queue2.list_posts()) == 1
    loaded = queue2.get_post(item.id)
    assert loaded is not None
    assert loaded.title == "Epic Tech Short"


def test_slot_spacing_and_daily_cap(tmp_path: Path):
    queue_file = tmp_path / "test_queue_slots.json"
    queue = PublishingQueue(queue_file=queue_file, post_gap_minutes=60, daily_cap=2)

    # Anchor now to 10:00 AM so p1 and p2 are guaranteed to fall on the same day regardless of execution time
    now = (dt.datetime.now() + dt.timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0).timestamp()

    # Post 1
    p1 = queue.enqueue(
        job_id="job_1",
        clip_id="c1",
        platform="tiktok",
        account_id="acc_tt",
        title="P1",
        video_path="p1.mp4",
        requested_time=now,
    )

    # Post 2: should be spaced by at least 60 minutes
    p2 = queue.enqueue(
        job_id="job_1",
        clip_id="c2",
        platform="tiktok",
        account_id="acc_tt",
        title="P2",
        video_path="p2.mp4",
    )

    assert p2.due_at >= p1.due_at + (60 * 60) - 1

    # Post 3: exceeds daily_cap of 2, so should roll over to tomorrow at 9:00 AM
    p3 = queue.enqueue(
        job_id="job_1",
        clip_id="c3",
        platform="tiktok",
        account_id="acc_tt",
        title="P3",
        video_path="p3.mp4",
    )

    lt3 = time.localtime(p3.due_at)
    assert lt3.tm_hour == 9
    assert lt3.tm_min == 0
    assert p3.due_at > p2.due_at


def test_mark_success_and_retry_backoff(tmp_path: Path):
    queue_file = tmp_path / "test_queue_retry.json"
    queue = PublishingQueue(queue_file=queue_file)

    item = queue.enqueue(
        job_id="j1",
        clip_id="c1",
        platform="instagram",
        account_id="acc_ig",
        title="IG Reel",
        video_path="reel.mp4",
        schedule_now=True,
    )

    queue.mark_posting(item.id)
    assert queue.get_post(item.id).status == "posting"
    assert queue.get_post(item.id).tries == 1

    # Mark failed with rate limit
    queue.mark_failed(item.id, "429 Rate Limit Exceeded", is_rate_limit=True)
    post = queue.get_post(item.id)
    assert post.status == "scheduled"
    assert post.next_retry_at >= time.time() + 290

    # Mark success
    queue.mark_success(item.id, remote_url="https://instagram.com/p/xyz", remote_id="xyz123")
    post = queue.get_post(item.id)
    assert post.status == "posted"
    assert post.remote_url == "https://instagram.com/p/xyz"
    assert post.remote_id == "xyz123"
    assert post.error_message is None


def test_cancel_and_retry(tmp_path: Path):
    queue_file = tmp_path / "test_queue_cancel.json"
    queue = PublishingQueue(queue_file=queue_file)

    item = queue.enqueue(
        job_id="j1",
        clip_id="c1",
        platform="youtube",
        account_id="acc_yt",
        title="Title",
        video_path="vid.mp4",
    )

    assert queue.cancel_post(item.id)
    assert queue.get_post(item.id).status == "cancelled"

    assert queue.retry_post(item.id)
    assert queue.get_post(item.id).status == "scheduled"
    assert queue.get_post(item.id).due_at <= time.time()
