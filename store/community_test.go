package store

import (
	"context"
	"testing"
)

func TestDetectCommunities_TwoClusters(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	// Cluster 1: alice — bob — carol (fully connected)
	insertEntity(t, db, "alice", "Alice", grp)
	insertEntity(t, db, "bob", "Bob", grp)
	insertEntity(t, db, "carol", "Carol", grp)
	insertEdge(t, db, "e1", "alice", "bob", grp)
	insertEdge(t, db, "e2", "bob", "carol", grp)
	insertEdge(t, db, "e3", "alice", "carol", grp)

	// Cluster 2: dave — eve (connected to each other, not to cluster 1)
	insertEntity(t, db, "dave", "Dave", grp)
	insertEntity(t, db, "eve", "Eve", grp)
	insertEdge(t, db, "e4", "dave", "eve", grp)

	result, err := db.DetectCommunities(ctx, grp, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	if result.Entities != 5 {
		t.Errorf("entities: want 5, got %d", result.Entities)
	}
	if result.Communities < 2 {
		t.Errorf("communities: want >= 2, got %d", result.Communities)
	}

	// alice and bob must be in the same community.
	aliceCom := db.EntityCommunityID(ctx, "alice")
	bobCom := db.EntityCommunityID(ctx, "bob")
	daveCom := db.EntityCommunityID(ctx, "dave")

	if aliceCom != bobCom {
		t.Errorf("alice (%d) and bob (%d) should be in same community", aliceCom, bobCom)
	}
	if aliceCom == daveCom {
		t.Errorf("alice (%d) and dave (%d) should be in different communities", aliceCom, daveCom)
	}
}

func TestDetectCommunities_SingleEntity(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	insertEntity(t, db, "alone", "Alone", grp)

	result, err := db.DetectCommunities(ctx, grp, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	if result.Entities != 1 {
		t.Errorf("entities: want 1, got %d", result.Entities)
	}
	// Single entity → no communities to detect.
	if result.Communities != 0 {
		t.Errorf("communities: want 0, got %d", result.Communities)
	}
}

func TestDetectCommunities_GroupIsolation(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	// Group A: alice — bob
	insertEntity(t, db, "alice", "Alice", "grpA")
	insertEntity(t, db, "bob", "Bob", "grpA")
	insertEdge(t, db, "e1", "alice", "bob", "grpA")

	// Group B: carol — dave
	insertEntity(t, db, "carol", "Carol", "grpB")
	insertEntity(t, db, "dave", "Dave", "grpB")
	insertEdge(t, db, "e2", "carol", "dave", "grpB")

	// Detect for group A only.
	result, err := db.DetectCommunities(ctx, "grpA", 1.0)
	if err != nil {
		t.Fatal(err)
	}

	if result.Entities != 2 {
		t.Errorf("entities: want 2, got %d", result.Entities)
	}

	// Carol (grpB) should not have been assigned.
	carolCom := db.EntityCommunityID(ctx, "carol")
	if carolCom != -1 {
		t.Errorf("carol community_id should be -1 (unassigned), got %d", carolCom)
	}
}

func TestEntitiesInCommunity(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	// Cluster: alice — bob — carol
	insertEntity(t, db, "alice", "Alice", grp)
	insertEntity(t, db, "bob", "Bob", grp)
	insertEntity(t, db, "carol", "Carol", grp)
	insertEdge(t, db, "e1", "alice", "bob", grp)
	insertEdge(t, db, "e2", "bob", "carol", grp)
	insertEdge(t, db, "e3", "alice", "carol", grp)

	if _, err := db.DetectCommunities(ctx, grp, 1.0); err != nil {
		t.Fatal(err)
	}

	aliceCom := db.EntityCommunityID(ctx, "alice")
	members, err := db.EntitiesInCommunity(ctx, grp, aliceCom)
	if err != nil {
		t.Fatal(err)
	}

	if len(members) != 3 {
		t.Errorf("community members: want 3, got %d", len(members))
	}
}
