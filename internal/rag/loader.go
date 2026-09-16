package rag

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cloudwego/eino/schema"
)

// LoadCorpus 把 dir 下的 .md 语料读成 Document 列表（跳过 SOURCES.md 这类版权说明）。
func LoadCorpus(ctx context.Context, dir string) ([]*schema.Document, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("读取语料目录 %s：%w", dir, err)
	}

	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") || e.Name() == "SOURCES.md" {
			continue
		}
		names = append(names, e.Name())
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("目录 %s 下没有可用的 .md 语料", dir)
	}
	sort.Strings(names)

	docs := make([]*schema.Document, 0, len(names))
	for _, name := range names {
		path := filepath.Join(dir, name)
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("读取 %s：%w", path, err)
		}
		docs = append(docs, &schema.Document{
			ID:       name,
			Content:  string(raw),
			MetaData: map[string]any{"source": filepath.ToSlash(path)},
		})
	}
	return docs, nil
}
