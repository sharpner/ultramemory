package store

import (
	"context"
	"database/sql"
	"fmt"
)

// JobType constants.
const (
	JobTypeIngest = "ingest"
)

// Job is a pending unit of work.
type Job struct {
	ID      int64
	Type    string
	Payload string // JSON
}

// PushJob enqueues a new job.
func (d *DB) PushJob(ctx context.Context, jobType, payload string) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT INTO jobs (type, payload) VALUES (?, ?)`,
		jobType, payload,
	)
	return err
}

// NextJob atomically claims the oldest pending job.
// Returns nil, nil when the queue is empty.
func (d *DB) NextJob(ctx context.Context) (*Job, error) {
	tx, err := d.sql.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return nil, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	var job Job
	err = tx.QueryRowContext(ctx,
		`SELECT id, type, payload FROM jobs
		 WHERE status = 'pending'
		 ORDER BY created_at ASC
		 LIMIT 1`,
	).Scan(&job.ID, &job.Type, &job.Payload)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("query job: %w", err)
	}

	_, err = tx.ExecContext(ctx,
		`UPDATE jobs
		 SET status = 'processing', updated_at = CURRENT_TIMESTAMP
		 WHERE id = ?`,
		job.ID,
	)
	if err != nil {
		return nil, fmt.Errorf("claim job: %w", err)
	}

	return &job, tx.Commit()
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
// If max_attempts exceeded the job stays failed; otherwise requeues as pending.
func (d *DB) FailJob(ctx context.Context, id int64, reason string) error {
	_, err := d.sql.ExecContext(ctx, `
		UPDATE jobs SET
			status     = CASE WHEN attempts + 1 >= max_attempts THEN 'failed' ELSE 'pending' END,
			attempts   = attempts + 1,
			error      = ?,
			updated_at = CURRENT_TIMESTAMP
		WHERE id = ?`,
		reason, id,
	)
	return err
}

// QueueStats returns pending/processing/done/failed counts.
func (d *DB) QueueStats(ctx context.Context) (map[string]int, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT status, count(*) FROM jobs GROUP BY status`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

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
