package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

// JobType constants.
const (
	JobTypeIngest = "ingest"
)

// Job is a pending unit of work.
type Job struct {
	ID       int64
	Type     string
	Payload  string // JSON
	Attempts int
}

// PushJob enqueues a new job.
func (d *DB) PushJob(ctx context.Context, jobType, payload string) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT INTO jobs (type, payload) VALUES (?, ?)`,
		jobType, payload,
	)
	return err
}

// NextJob atomically claims the oldest pending job using UPDATE...RETURNING.
// Single statement — no transaction needed, minimal lock contention.
// Returns nil, nil when the queue is empty.
func (d *DB) NextJob(ctx context.Context) (*Job, error) {
	var job Job
	err := d.sql.QueryRowContext(ctx, `
		UPDATE jobs
		SET status = 'processing', updated_at = CURRENT_TIMESTAMP
		WHERE id = (
			SELECT id FROM jobs
			WHERE status = 'pending'
			  AND (not_before IS NULL OR not_before <= CURRENT_TIMESTAMP)
			ORDER BY created_at ASC
			LIMIT 1
		)
		RETURNING id, type, payload, attempts`,
	).Scan(&job.ID, &job.Type, &job.Payload, &job.Attempts)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("claim job: %w", err)
	}
	return &job, nil
}

// CompleteJob marks a job as done.
func (d *DB) CompleteJob(ctx context.Context, id int64) error {
	_, err := d.sql.ExecContext(ctx,
		`UPDATE jobs SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE id = ?`,
		id,
	)
	return err
}

// FailJob marks a job as failed and increments attempts.
// If max_attempts exceeded the job stays failed; otherwise requeues as pending
// with an exponential backoff delay (5s → 15s → 30s) before it becomes eligible again.
func (d *DB) FailJob(ctx context.Context, id int64, reason string) error {
	_, err := d.sql.ExecContext(ctx, `
		UPDATE jobs SET
			status     = CASE WHEN attempts + 1 >= max_attempts THEN 'failed' ELSE 'pending' END,
			attempts   = attempts + 1,
			error      = ?,
			updated_at = CURRENT_TIMESTAMP,
			not_before = CASE
				WHEN attempts + 1 >= max_attempts THEN NULL
				WHEN attempts = 0 THEN datetime('now', '+5 seconds')
				WHEN attempts = 1 THEN datetime('now', '+15 seconds')
				ELSE datetime('now', '+30 seconds')
			END
		WHERE id = ?`,
		reason, id,
	)
	return err
}

// RecoverStaleJobs resets jobs stuck in 'processing' longer than staleAfter
// back to 'pending'. This recovers from worker crashes where FailJob was never called.
func (d *DB) RecoverStaleJobs(ctx context.Context, staleAfter time.Duration) (int64, error) {
	cutoff := time.Now().Add(-staleAfter).UTC().Format("2006-01-02 15:04:05")
	res, err := d.sql.ExecContext(ctx, `
		UPDATE jobs
		SET status = 'pending', updated_at = CURRENT_TIMESTAMP
		WHERE status = 'processing' AND updated_at < ?`,
		cutoff,
	)
	if err != nil {
		return 0, fmt.Errorf("recover stale jobs: %w", err)
	}
	return res.RowsAffected()
}

// RequeueFailed resets all failed jobs back to pending with attempts zeroed.
func (d *DB) RequeueFailed(ctx context.Context) (int64, error) {
	res, err := d.sql.ExecContext(ctx, `
		UPDATE jobs
		SET status = 'pending', attempts = 0, error = NULL, updated_at = CURRENT_TIMESTAMP
		WHERE status = 'failed'`,
	)
	if err != nil {
		return 0, fmt.Errorf("requeue failed: %w", err)
	}
	return res.RowsAffected()
}

// QueueStats returns pending/processing/done/failed counts.
func (d *DB) QueueStats(ctx context.Context) (map[string]int, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT status, count(*) FROM jobs GROUP BY status`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	stats := map[string]int{}
	for rows.Next() {
		var status string
		var count int
		if err := rows.Scan(&status, &count); err != nil {
			return nil, err
		}
		stats[status] = count
	}
	return stats, rows.Err()
}
