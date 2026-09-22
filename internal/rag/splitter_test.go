package rag

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/cloudwego/eino/schema"
)

var sample = strings.Join([]string{
	"# 记忆检索",
	"",
	"前言段落。",
	"",
	"## 1 向量检索",
	"",
	"第一段。",
	"",
	"```python",
	"# 这是注释，不是标题",
	"def f():",
	"    return 1",
	"```",
	"",
	"### 1.1 细节",
	"",
	"| 列A | 列B |",
	"| --- | --- |",
	"| 1 | 2 |",
	"",
	"结尾段。",
}, "\n")

// 代码围栏里的 # 注释不能被当成标题，且整个代码块应完整落在同一块里。
func TestSplitMarkdownFenceIsNotHeading(t *testing.T) {
	pieces := SplitMarkdown(sample, 800)
	for _, p := range pieces {
		for _, h := range p.HeadingPath {
			if strings.Contains(h, "这是注释") {
				t.Fatalf("围栏内的 # 被当成了标题：%v", p.HeadingPath)
			}
		}
	}
	var code *Piece
	for i, p := range pieces {
		if strings.Contains(p.Body, "def f():") {
			code = &pieces[i]
		}
	}
	if code == nil {
		t.Fatal("没找到含代码块的块")
	}
	for _, want := range []string{"```python", "# 这是注释，不是标题", "```"} {
		if !strings.Contains(code.Body, want) {
			t.Errorf("代码块被切散了，缺 %q：\n%s", want, code.Body)
		}
	}
}

// 标题栈：A → A/B → A/B/C；回到上层后继任（A/D）。
func TestSplitMarkdownHeadingStack(t *testing.T) {
	pieces := SplitMarkdown(sample, 800)
	foundDeep := false
	for _, p := range pieces {
		path := strings.Join(p.HeadingPath, " > ")
		if path == "记忆检索 > 1 向量检索 > 1.1 细节" {
			foundDeep = true
		}
		if strings.Contains(p.Body, "前言段落") {
			if path != "记忆检索" {
				t.Errorf("前言所在节标题链应为 [记忆检索]，got %v", p.HeadingPath)
			}
		}
	}
	if !foundDeep {
		t.Errorf("没有出现三级标题链，pieces=%d", len(pieces))
	}

	tail := SplitMarkdown(sample+"\n\n# A\n\nx\n\n## B\n\ny\n\n## D\n\nz", 800)
	var last string
	for _, p := range tail {
		if strings.Contains(p.Body, "z") {
			last = strings.Join(p.HeadingPath, " > ")
		}
	}
	if last != "A > D" {
		t.Errorf("同级标题应替换栈顶，want [A > D]，got %q", last)
	}
}

// 关键不变量：每块正文与源文档区间逐字对应。
func TestSplitMarkdownExactSpans(t *testing.T) {
	runes := []rune(sample)
	pieces := SplitMarkdown(sample, 800)
	if len(pieces) == 0 {
		t.Fatal("没切出任何块")
	}
	for _, p := range pieces {
		end := p.Start + utf8.RuneCountInString(p.Body)
		if end > len(runes) {
			t.Fatalf("区间越界：%d > %d", end, len(runes))
		}
		if got := string(runes[p.Start:end]); got != p.Body {
			t.Errorf("Body 与原文区间不一致：\nstart=%d\nbody=%q\n原文=%q", p.Start, p.Body, got)
		}
	}
}

// 硬上限：超长单元（无换行的长段落）兜底切分后仍 ≤ maxLen，且拼起来与原文一致。
func TestSplitMarkdownOversizeFallback(t *testing.T) {
	long := strings.Repeat("这是一句很长的正文。", 300) // 3000 字
	pieces := SplitMarkdown("# 标题\n\n"+long, 800)
	if len(pieces) < 4 {
		t.Fatalf("超长单元应被切至少 4 块，got %d", len(pieces))
	}
	var joined strings.Builder
	for _, p := range pieces {
		if n := utf8.RuneCountInString(p.Body); n > 800 {
			t.Errorf("块超过 maxLen：%d 字", n)
		}
		joined.WriteString(p.Body)
	}
	if !strings.Contains(joined.String(), long) {
		t.Error("兜底切分后有内容丢失")
	}
}

// 不产生只含空白的垃圾块。
func TestSplitMarkdownNoBlankOnlyPiece(t *testing.T) {
	for _, p := range SplitMarkdown(sample, 800) {
		if strings.TrimSpace(p.Body) == "" {
			t.Errorf("出现纯空白块：start=%d", p.Start)
		}
	}
}

// 面包屑消融：heading-nobc 的正文不带 【标题链】 前缀，但 meta 里仍保留 heading_path。
func TestHeadingNoBreadcrumb(t *testing.T) {
	docs := []*schema.Document{{ID: "d.md", Content: sample}}
	run := func(name string) []*schema.Document {
		tf, err := NewTransformer(name, 500, 50, 800)
		if err != nil {
			t.Fatal(err)
		}
		chunks, err := tf.Transform(context.Background(), docs)
		if err != nil {
			t.Fatal(err)
		}
		if len(chunks) == 0 {
			t.Fatalf("%s 没切出块", name)
		}
		return chunks
	}

	for _, c := range run("heading") {
		if !strings.HasPrefix(c.Content, "【") {
			t.Errorf("heading 的正文应带标题链前缀：%q", c.Content)
		}
	}

	paths := 0
	for _, c := range run("heading-nobc") {
		if strings.HasPrefix(c.Content, "【") {
			t.Errorf("heading-nobc 的正文不应带前缀：%q", c.Content)
		}
		if c.MetaData["strategy"] != "heading-nobc" {
			t.Errorf("strategy meta = %v，want heading-nobc", c.MetaData["strategy"])
		}
		if p, _ := c.MetaData["heading_path"].(string); p != "" {
			paths++
		}
	}
	if paths == 0 {
		t.Error("heading-nobc 丢了 meta 里的 heading_path")
	}
}

// 真实语料上的地基不变量：所有块的 Body 与源文档区间逐字对应（覆盖率评测依赖它）。
func TestCorpusSplitMarkdownExactSpans(t *testing.T) {
	docs, err := LoadCorpus(context.Background(), "../../corpus")
	if err != nil {
		t.Skipf("跳过：读不到语料目录（%v）", err)
	}
	total := 0
	for _, doc := range docs {
		runes := []rune(doc.Content)
		for _, p := range SplitMarkdown(doc.Content, 800) {
			total++
			end := p.Start + utf8.RuneCountInString(p.Body)
			if end > len(runes) || string(runes[p.Start:end]) != p.Body {
				t.Fatalf("%s：区间失配 start=%d end=%d（全文 %d 字）", doc.ID, p.Start, end, len(runes))
			}
		}
	}
	t.Logf("语料 %d 篇，块 %d 个，区间全部精确", len(docs), total)
}
