package store

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
)

func BenchmarkUpsertEntity(b *testing.B) {
	db := openTestDB(b)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.UpsertEntity(ctx, Entity{
			UUID:       fmt.Sprintf("e%d", i),
			Name:       fmt.Sprintf("Entity Number %d", i),
			EntityType: "person",
			GroupID:    "g",
		})
	}
}

func BenchmarkSearchEntitiesFTS(b *testing.B) {
	db := openTestDB(b)
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		_, _ = db.UpsertEntity(ctx, Entity{
			UUID:       fmt.Sprintf("e%d", i),
			Name:       fmt.Sprintf("Person %d", i),
			EntityType: "person",
			GroupID:    "g",
		})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.SearchEntitiesFTS(ctx, "Person", "g", 10)
	}
}

// BenchmarkConcurrentReadWhileWriting verifies reads proceed during writes (WAL).
// "reads/write" > 0 confirms readers are never blocked by the writer.
func BenchmarkConcurrentReadWhileWriting(b *testing.B) {
	db := openTestDB(b)
	ctx := context.Background()

	for i := 0; i < 50; i++ {
		_, _ = db.UpsertEntity(ctx, Entity{
			UUID:       fmt.Sprintf("seed%d", i),
			Name:       fmt.Sprintf("Seed Person %d", i),
			EntityType: "person",
			GroupID:    "g",
		})
	}

	stop := make(chan struct{})
	var readCount atomic.Int64

	for r := 0; r < 4; r++ {
		go func() {
			for {
				select {
				case <-stop:
					return
				default:
					_, _ = db.SearchEntitiesFTS(ctx, "Seed", "g", 10)
					readCount.Add(1)
				}
			}
		}()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.UpsertEntity(ctx, Entity{
			UUID:       fmt.Sprintf("w%d", i),
			Name:       fmt.Sprintf("Writer %d", i),
			EntityType: "person",
			GroupID:    "g",
		})
	}
	b.StopTimer()

	close(stop)
	reads := readCount.Load()
	if b.N > 0 {
		b.ReportMetric(float64(reads)/float64(b.N), "reads/write")
	}
}
