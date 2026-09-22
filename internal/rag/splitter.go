package rag

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/schema"
)

var _ document.Transformer = (*Splitter)(nil)
var _ document.Transformer = (*HeadingSplitter)(nil)

// NewTransformer 按名字构造切块器（2B 对比实验入口）：
//
//	fixed        —— 定长 + 重叠
//	heading      —— 结构感知（按标题分节，节内按自然单元打包），正文带标题链前缀
//	heading-nobc —— 同 heading，但标题链只进 meta 不进正文（面包屑消融实验）
func NewTransformer(name string, size, overlap, maxLen int) (document.Transformer, error) {
	switch name {
	case "fixed":
		return &Splitter{ChunkSize: size, Overlap: overlap}, nil
	case "heading":
		return &HeadingSplitter{MaxLen: maxLen}, nil
	case "heading-nobc":
		return &HeadingSplitter{MaxLen: maxLen, NoBreadcrumb: true}, nil
	default:
		return nil, fmt.Errorf("未知切块策略 %q（可选：fixed / heading / heading-nobc）", name)
	}
}

// chunkMeta 组装块的元数据：复制来源 meta，再写入切块自述信息。
// start/end 是块"正文"在源文档里的 rune 区间 —— 评测算覆盖率时靠它把块映射回原文。
// 注意：content 里若有拼上去的前缀（如标题链），不计入区间。
func chunkMeta(doc *schema.Document, idx int, strategy string, start, end int, extra map[string]any) map[string]any {
	meta := make(map[string]any, len(doc.MetaData)+5+len(extra))
	for k, v := range doc.MetaData {
		meta[k] = v
	}
	meta["doc"] = doc.ID
	meta["chunk_index"] = idx
	meta["strategy"] = strategy
	meta["start"] = start
	meta["end"] = end
	for k, v := range extra {
		meta[k] = v
	}
	return meta
}

// Splitter 定长 + 重叠切块器，实现 Eino 的 document.Transformer 接口。
type Splitter struct {
	ChunkSize int // 每块目标长度（rune 数）
	Overlap   int // 相邻块重叠长度（rune 数）
}

func (s *Splitter) Transform(ctx context.Context, src []*schema.Document, opts ...document.TransformerOption) ([]*schema.Document, error) {
	var out []*schema.Document
	step := s.ChunkSize - s.Overlap
	for _, doc := range src {
		for i, piece := range SplitText(doc.Content, s.ChunkSize, s.Overlap) {
			start := i * step // 与 SplitText 内部一致：第 i 块从 i*step 起
			out = append(out, &schema.Document{
				ID:       fmt.Sprintf("%s#%d", doc.ID, i),
				Content:  piece,
				MetaData: chunkMeta(doc, i, "fixed", start, start+utf8.RuneCountInString(piece), nil),
			})
		}
	}
	return out, nil
}

// SplitText 把一段文本切成块 —— 步骤 3 由你实现。
//
// 要求：
//  1. 按 rune（字符）切，不要按 byte（中文一个字占 3 个字节，按 byte 切会出现乱码）
//  2. 步长 = chunkSize - overlap：相邻两块共享 overlap 个字符
//  3. 最后一块不足 chunkSize 也要保留（除非为空）
//  4. chunkSize <= 0、overlap < 0、overlap >= chunkSize 时返回空切片
func SplitText(text string, chunkSize, overlap int) []string {
	if chunkSize <= 0 || overlap < 0 || overlap >= chunkSize {
		// fmt.Println("Invalid chunkSize or overlap")
		return nil
	}
	step := chunkSize - overlap
	var chunks []string
	runes := []rune(text)
	var subrunes []rune
	for i := 0; i < len(runes); i += step {
		subrunes = subrunes[:0] // 清空子切片
		for j := i; j < i+chunkSize && j < len(runes); j++ {
			subrunes = append(subrunes, runes[j])
		}
		chunks = append(chunks, string(subrunes))

		//程序可以两写完
		//end := min(i+chunkSize, len(runes))
		//chunks = append(chunks, string(runes[i:end]))

	}
	//panic("SplitText 尚未实现")
	return chunks
}

// Piece 结构感知切块的最小输出单元。
type Piece struct {
	HeadingPath []string // 从 H1 到当前节的标题链（已去掉 # 号）；文档开头、标题之前的正文为空
	Body        string   // 正文（不含标题行与标题链前缀；单元之间的空行保留，正文与源文档逐字对应）
	Start       int      // Body 第一个字符在源文档里的 rune 下标
}

// SplitMarkdown 结构感知切块 —— 阶段 2B 核心任务 1，由你实现。
//
// 输入：一篇 markdown 的全文；maxLen：单块正文的 rune 上限。
// 输出：按文档顺序排列的 Piece 列表。
//
// 规则：
//  1. 标题行 = ^#{1,4} 开头且后跟空格的行（如 "## 1.2 什么是 RAG"），但 ``` 代码围栏内的 # 行
//     不是标题 —— 语料里代码块很多，Python/shell 注释全以 # 开头，不跟踪围栏开/关会产生大量假标题。
//  2. HeadingPath 按层级维护（想象一个栈）：遇到更深标题压栈；遇到更浅或同级先弹到合适层级再压。
//     例：# A → [A]；## B → [A B]；### C → [A B C]；## D → [A D]。
//  3. 节内按"自然单元"贪心打包成 ≤ maxLen 的块，不要从单元内部切断：
//     自然单元 = 段落（空行分隔）｜ 整个代码块（```...```）｜ 整张表（连续的 | 行）。
//     只有当单元自身就超过 maxLen 时，才允许在单元内按行兜底切。
//  4. 标题行本身不进 Body（标题信息由 HeadingSplitter 拼前缀表达）；Body 为空的节不出块；
//     节与节之间不重叠；每节最后一块可以不满。
//  5. Start 必须保真：Body 首字符在全文中的 rune 下标（提示：行首下标 = 前面所有行的 rune 数累加）。
//
// 建议步骤：先把全文按行切开并标注每行角色（标题 / 围栏内 / 表格行 / 正文），
// 再聚合成单元列表，最后贪心打包。写完先用 go run ./cmd/chunk -strategy heading 预览。
func SplitMarkdown(text string, maxLen int) []Piece {
	if maxLen <= 0 {
		maxLen = 800
	}
	runes := []rune(text)
	lines := strings.Split(text, "\n")

	// starts[i] = 第 i 行行首在全文中的 rune 下标；total = 全文长度
	starts := make([]int, len(lines))
	total := 0
	for i, line := range lines {
		starts[i] = total
		total += utf8.RuneCountInString(line)
		if i < len(lines)-1 {
			total++ // 行尾换行符
		}
	}

	var (
		pieces []Piece
		stack  []headingNode
		units  []mdUnit
	)

	// addUnit 把第 i 行到第 j-1 行（含行尾换行）收为一个自然单元。
	addUnit := func(i, j int) {
		end := total
		if j < len(lines) {
			end = starts[j]
		}
		units = append(units, mdUnit{text: string(runes[starts[i]:end]), start: starts[i]})
	}
	// flush 把当前积累的单元按当前标题链打包，收成一节的块。
	flush := func() {
		if len(units) == 0 {
			return
		}
		path := make([]string, len(stack))
		for i, h := range stack {
			path[i] = h.title
		}
		pieces = append(pieces, packUnits(units, maxLen, path)...)
		units = nil
	}

	for i := 0; i < len(lines); {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		// 代码块：从开围栏一路吃到闭围栏（没闭合就吃到文末），整块一个单元，不进块内部分类
		if strings.HasPrefix(trimmed, "```") {
			j := i + 1
			for j < len(lines) && !strings.HasPrefix(strings.TrimSpace(lines[j]), "```") {
				j++
			}
			if j < len(lines) {
				j++ // 收下闭合围栏那一行
			}
			addUnit(i, j)
			i = j
			continue
		}

		// 标题行：只更新标题栈，本身不进 Body
		if hn, ok := headingInfo(line); ok {
			flush()
			for len(stack) > 0 && stack[len(stack)-1].level >= hn.level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, hn)
			i++
			continue
		}

		// 空行：单独成单元，保证 Body 与源文档逐字对应（不留空洞）
		if trimmed == "" {
			addUnit(i, i+1)
			i++
			continue
		}

		// 表格：连续的 | 行整张一个单元
		if strings.HasPrefix(trimmed, "|") {
			j := i + 1
			for j < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[j]), "|") {
				j++
			}
			addUnit(i, j)
			i = j
			continue
		}

		// 普通段落：连续吃到空行 / 标题 / 表格 / 围栏为止
		j := i + 1
		for j < len(lines) {
			t := strings.TrimSpace(lines[j])
			if t == "" || strings.HasPrefix(t, "|") || strings.HasPrefix(t, "```") {
				break
			}
			if _, ok := headingInfo(lines[j]); ok {
				break
			}
			j++
		}
		addUnit(i, j)
		i = j
	}
	flush()
	return pieces
}

// mdUnit 一个自然单元：单元内文本在源文档中连续，start 是它在全文里的 rune 起始下标。
// text 末尾带与下一行之间的换行符（文末单元除外），保证单元首尾相接后与原文逐字一致。
type mdUnit struct {
	text  string
	start int
}

// headingNode 标题栈的一层。
type headingNode struct {
	level int
	title string
}

// headingInfo 判断一行是否为标题：1~4 个 # 开头，且后面紧跟空格/制表符（"#标签"不算）。
func headingInfo(line string) (headingNode, bool) {
	lv := 0
	for lv < len(line) && lv < 4 && line[lv] == '#' {
		lv++
	}
	if lv == 0 || lv >= len(line) || (line[lv] != ' ' && line[lv] != '\t') {
		return headingNode{}, false
	}
	title := strings.TrimSpace(line[lv:])
	if title == "" {
		return headingNode{}, false
	}
	return headingNode{level: lv, title: title}, true
}

// expandUnits 预处理超长单元：单元自身超过 maxLen 时按行拆小，单行还超长就按 maxLen 硬切。
// 普通单元原样保留 —— 主力是打包阶段的贪心合并，这里只兜底。
func expandUnits(units []mdUnit, maxLen int) []mdUnit {
	var out []mdUnit
	for _, u := range units {
		r := []rune(u.text)
		if len(r) <= maxLen {
			out = append(out, u)
			continue
		}
		for pos := 0; pos < len(r); {
			end := pos
			for end < len(r) && r[end] != '\n' {
				end++
			}
			if end < len(r) {
				end++ // 带上行尾换行
			}
			for s := pos; s < end; s += maxLen {
				out = append(out, mdUnit{text: string(r[s:min(s+maxLen, end)]), start: u.start + s})
			}
			pos = end
		}
	}
	return out
}

// packUnits 把一节内的自然单元按顺序贪心打包成 ≤ maxLen 的块（不切断单元，除非单元超限）。
func packUnits(units []mdUnit, maxLen int, path []string) []Piece {
	units = expandUnits(units, maxLen)

	var (
		out      []Piece
		buf      strings.Builder
		curLen   int
		curStart int
	)
	emit := func() {
		if curLen == 0 {
			return
		}
		out = append(out, Piece{HeadingPath: path, Body: buf.String(), Start: curStart})
		buf.Reset()
		curLen = 0
	}

	for _, u := range units {
		n := utf8.RuneCountInString(u.text)
		if curLen == 0 && strings.TrimSpace(u.text) == "" {
			continue // 纯空行不开新块（否则标题后的空行会变成只含空白的垃圾块）
		}
		if curLen > 0 && curLen+n > maxLen {
			emit()
		}
		if curLen == 0 {
			curStart = u.start
		}
		buf.WriteString(u.text)
		curLen += n
	}
	emit()
	return out
}

// HeadingSplitter 结构感知切块器：SplitMarkdown 负责切，这里负责组装成 Eino Document。
// 默认给正文拼上标题链前缀 【A > B > C】—— 让向量和 BM25 都看得见"这块属于哪一节"；
// NoBreadcrumb=true 时标题链只写 meta["heading_path"]，不进正文（面包屑消融实验）。
type HeadingSplitter struct {
	MaxLen       int  // 单块正文上限（rune），<= 0 时默认 800
	NoBreadcrumb bool // true = 标题链不进正文
}

func (h *HeadingSplitter) Transform(ctx context.Context, src []*schema.Document, opts ...document.TransformerOption) ([]*schema.Document, error) {
	maxLen := h.MaxLen
	if maxLen <= 0 {
		maxLen = 800
	}
	strategy := "heading"
	if h.NoBreadcrumb {
		strategy = "heading-nobc"
	}
	var out []*schema.Document
	for _, doc := range src {
		for i, p := range SplitMarkdown(doc.Content, maxLen) {
			content := p.Body
			extra := map[string]any{}
			if len(p.HeadingPath) > 0 {
				path := strings.Join(p.HeadingPath, " > ")
				if !h.NoBreadcrumb {
					content = "【" + path + "】\n\n" + p.Body
				}
				extra["heading_path"] = path
			}
			out = append(out, &schema.Document{
				ID:       fmt.Sprintf("%s#%d", doc.ID, i),
				Content:  content,
				MetaData: chunkMeta(doc, i, strategy, p.Start, p.Start+utf8.RuneCountInString(p.Body), extra),
			})
		}
	}
	return out, nil
}
