package store

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"
)

func openQueueTestDB(t *testing.T) *DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "queue-test-*.db")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	_ = f.Close()
	db, err := Open(f.Name())
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestNextJob_Atomic(t *testing.T) {
	db := openQueueTestDB(t)
	ctx := context.Background()

	// Push 10 jobs.
	for i := 0; i < 10; i++ {
		if err := db.PushJob(ctx, JobTypeIngest, `{"i":`+string(rune('0'+i))+`}`); err != nil {
			t.Fatal(err)
		}
	}

	// Claim all 10 from multiple goroutines — each job must be claimed exactly once.
	var mu sync.Mutex
	claimed := map[int64]bool{}
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				job, err := db.NextJob(ctx)
				if err != nil {
					t.Error(err)
					return
				}
				if job == nil {
					return
				}
				mu.Lock()
				if claimed[job.ID] {
					t.Errorf("job %d claimed twice", job.ID)
				}
				claimed[job.ID] = true
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if len(claimed) != 10 {
		t.Errorf("expected 10 unique jobs claimed, got %d", len(claimed))
	}
}

func TestNextJob_EmptyQueue(t *testing.T) {
	db := openQueueTestDB(t)
	job, err := db.NextJob(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if job != nil {
		t.Errorf("expected nil job from empty queue, got %+v", job)
	}
}

func TestRecoverStaleJobs(t *testing.T) {
	db := openQueueTestDB(t)
	ctx := context.Background()

	// Push and claim a job.
	if err := db.PushJob(ctx, JobTypeIngest, `{"test":1}`); err != nil {
		t.Fatal(err)
	}
	job, err := db.NextJob(ctx)
	if err != nil || job == nil {
		t.Fatal("expected a job")
	}

	// Backdate updated_at to simulate a stale job.
	_, err = db.sql.ExecContext(ctx,
		`UPDATE jobs SET updated_at = datetime('now', '-10 minutes') WHERE id = ?`,
		job.ID,
	)
	if err != nil {
		t.Fatal(err)
	}

	// Recovery with 5min threshold should find it.
	n, err := db.RecoverStaleJobs(ctx, 5*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("expected 1 recovered job, got %d", n)
	}

	// Job should be claimable again.
	recovered, err := db.NextJob(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if recovered == nil {
		t.Fatal("expected recovered job to be claimable")
	}
	if recovered.ID != job.ID {
		t.Errorf("expected job ID %d, got %d", job.ID, recovered.ID)
	}
}

func TestRecoverStaleJobs_FreshJobNotRecovered(t *testing.T) {
	db := openQueueTestDB(t)
	ctx := context.Background()

	// Push and claim — but DON'T backdate.
	if err := db.PushJob(ctx, JobTypeIngest, `{"test":1}`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.NextJob(ctx); err != nil {
		t.Fatal(err)
	}

	// Recovery should find nothing — job was just claimed.
	n, err := db.RecoverStaleJobs(ctx, 5*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("expected 0 recovered jobs for fresh processing job, got %d", n)
	}
}

func TestCompleteAndFailJob(t *testing.T) {
	db := openQueueTestDB(t)
	ctx := context.Background()

	// Push two jobs.
	if err := db.PushJob(ctx, JobTypeIngest, `{"a":1}`); err != nil {
		t.Fatal(err)
	}
	if err := db.PushJob(ctx, JobTypeIngest, `{"b":2}`); err != nil {
		t.Fatal(err)
	}

	// Claim and complete first.
	job1, _ := db.NextJob(ctx)
	if err := db.CompleteJob(ctx, job1.ID); err != nil {
		t.Fatal(err)
	}

	// Claim and fail second.
	job2, _ := db.NextJob(ctx)
	if err := db.FailJob(ctx, job2.ID, "test error"); err != nil {
		t.Fatal(err)
	}

	stats, err := db.QueueStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats["done"] != 1 {
		t.Errorf("expected 1 done, got %d", stats["done"])
	}
	// FailJob requeues as pending if attempts < max_attempts.
	if stats["pending"] != 1 {
		t.Errorf("expected 1 pending (requeued), got %d", stats["pending"])
	}
}
