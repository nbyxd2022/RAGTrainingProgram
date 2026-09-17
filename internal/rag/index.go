package rag

import (
	"context"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/schema"
)

// IndexEntry 索引里的一条：块的原文、元数据和向量。
// 向量存 float32：2048 维省一半磁盘，精度对相似度排序没有影响。
type IndexEntry struct {
	ID       string
	Content  string
	MetaData map[string]any
	Vector   []float32
}

type VectorIndex struct {
	Entries []IndexEntry
}

// BuildIndex 把所有块交给 embedding 模型，构建向量索引。
// 多模态接口一次只接受一条文本，所以这里按批提交，靠适配器内部的并发（5 路）跑满。
func BuildIndex(ctx context.Context, chunks []*schema.Document, embedder embedding.Embedder, onProgress func(done, total int)) (*VectorIndex, error) {
	const batchSize = 20

	idx := &VectorIndex{Entries: make([]IndexEntry, 0, len(chunks))}
	for start := 0; start < len(chunks); start += batchSize {
		end := min(start+batchSize, len(chunks))
		batch := chunks[start:end]

		texts := make([]string, len(batch))
		for i, c := range batch {
			texts[i] = c.Content
		}
		vecs, err := embedder.EmbedStrings(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("第 %d~%d 块向量化失败：%w", start, end-1, err)
		}
		if len(vecs) != len(batch) {
			return nil, fmt.Errorf("向量化返回 %d 条，输入 %d 条", len(vecs), len(batch))
		}

		for i, c := range batch {
			idx.Entries = append(idx.Entries, IndexEntry{
				ID:       c.ID,
				Content:  c.Content,
				MetaData: c.MetaData,
				Vector:   ToFloat32(vecs[i]),
			})
		}
		if onProgress != nil {
			onProgress(end, len(chunks))
		}
	}
	return idx, nil
}

func (idx *VectorIndex) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(idx)
}

func LoadIndex(path string) (*VectorIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var idx VectorIndex
	if err := gob.NewDecoder(f).Decode(&idx); err != nil {
		return nil, err
	}
	return &idx, nil
}

// ToDocuments 把索引条目还原成文档，供 BM25 等其他检索器消费。
// 同一份块喂给两条检索路线，ID 才能对齐、RRF 才能融合。
func (idx *VectorIndex) ToDocuments() []*schema.Document {
	out := make([]*schema.Document, len(idx.Entries))
	for i, e := range idx.Entries {
		out[i] = &schema.Document{ID: e.ID, Content: e.Content, MetaData: e.MetaData}
	}
	return out
}

func ToFloat32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}
