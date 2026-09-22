package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/cloudwego/eino/schema"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

// sections 是评测集标注工具：按标题导出"节内容区间"、把引文定位成精确 rune 区间、
// 按区间回看原文，或把标注 spec 编译成 testset v2 JSON。
// 用途：把 testset 的 gold 从"某个索引的块 ID"升级成"源文档字符区间"——与切块策略无关。
func main() {
	doc := flag.String("doc", "", "只处理 ID 含该子串的文档（如 14 或 memory）")
	find := flag.String("find", "", "在语料中查找该引文（要求精确子串），输出文档与 [start,end) 区间")
	spans := flag.String("spans", "", "配合 -doc：回看这些 rune 区间 [start:end) 的原文（逗号分隔，如 0:500,450:1400）")
	emit := flag.String("emit", "", "读标注 spec 文件（===Q 问题 / tags: / note: / ===S 引文段），编译成 testset v2 JSON")
	out := flag.String("out", "", "配合 -emit：JSON 写入该文件（默认打印到 stdout）")
	previewN := flag.Int("preview", 70, "章节预览字数（0 = 只列区间）")
	pad := flag.Int("pad", 60, "-spans 模式：区间前后各附多少个字做上下文")
	flag.Parse()

	ctx := context.Background()
	docs, err := rag.LoadCorpus(ctx, "corpus")
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case *emit != "":
		emitTestset(docs, *emit, *out)
	case *spans != "":
		showSpans(docs, *doc, *spans, *pad)
	case *find != "":
		findQuote(docs, *doc, *find)
	default:
		listSections(docs, *doc, *previewN)
	}
}

// listSections 列出每篇文档"按标题分节"的区间清单。
// 技巧：maxLen 拉到极大，SplitMarkdown 的贪心装箱会把每节所有单元合成一个 Piece，
// [Start, Start+len(Body)) 就是该节内容在源文档里的精确区间（标题行本身不在其中）。
func listSections(docs []*schema.Document, filter string, previewN int) {
	total := 0
	for _, d := range docs {
		if filter != "" && !strings.Contains(d.ID, filter) {
			continue
		}
		pieces := rag.SplitMarkdown(d.Content, 1<<30)
		fmt.Printf("== %s（%d 字，%d 节）==\n", d.ID, utf8.RuneCountInString(d.Content), len(pieces))
		for _, p := range pieces {
			end := p.Start + utf8.RuneCountInString(p.Body)
			path := strings.Join(p.HeadingPath, " > ")
			if path == "" {
				path = "（标题前的正文）"
			}
			fmt.Printf("  [%6d, %6d) %5d 字  %s\n", p.Start, end, end-p.Start, path)
			if previewN > 0 {
				fmt.Printf("        %s\n", preview(p.Body, previewN))
			}
		}
		total += len(pieces)
		fmt.Println()
	}
	fmt.Printf("共 %d 节。标注提示：用 -find \"原文引文\" 拿精确区间。\n", total)
}

// showSpans 按 rune 区间回看原文：重标 gold 时先看旧 gold 块实际覆盖了哪些文字，
// 再决定"最小充分答案"该截到哪一句。
func showSpans(docs []*schema.Document, filter, spec string, pad int) {
	var ranges [][2]int
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		ab := strings.SplitN(part, ":", 2)
		if len(ab) != 2 {
			log.Fatalf("区间格式应为 start:end（rune 下标），收到 %q", part)
		}
		a, e1 := strconv.Atoi(strings.TrimSpace(ab[0]))
		b, e2 := strconv.Atoi(strings.TrimSpace(ab[1]))
		if e1 != nil || e2 != nil {
			log.Fatalf("区间解析失败：%q", part)
		}
		ranges = append(ranges, [2]int{a, b})
	}

	matched := 0
	for _, d := range docs {
		if filter != "" && !strings.Contains(d.ID, filter) {
			continue
		}
		matched++
		runes := []rune(d.Content)
		fmt.Printf("== %s（%d 字）==\n", d.ID, len(runes))
		for _, r := range ranges {
			a, b := max(0, r[0]), min(len(runes), r[1])
			if a >= b {
				fmt.Printf("  [%d,%d)：越界或为空\n", r[0], r[1])
				continue
			}
			fmt.Printf("  [%d,%d)（%d 字）\n%s\n\n", a, b, b-a, contextAround(runes, a, b, pad))
		}
	}
	if matched == 0 {
		fmt.Println("没有匹配的文档（检查 -doc 过滤词）")
	}
}

// findQuote 精确定位一段引文，打印它的 [start,end) 与上下文。
// 标注 gold 的流程：先在这里定位，再决定是否把区间扩到整个自然段。
func findQuote(docs []*schema.Document, filter, quote string) {
	hits := 0
	for _, d := range docs {
		if filter != "" && !strings.Contains(d.ID, filter) {
			continue
		}
		content := d.Content
		for from := 0; from < len(content); {
			i := strings.Index(content[from:], quote)
			if i < 0 {
				break
			}
			byteOff := from + i
			start := utf8.RuneCountInString(content[:byteOff])
			end := start + utf8.RuneCountInString(quote)
			fmt.Printf("%s [%d,%d)\n    %s\n", d.ID, start, end, contextAround([]rune(content), start, end, 40))
			hits++
			from = byteOff + len(quote)
		}
	}
	if hits == 0 {
		fmt.Println("没找到：要求精确子串（全半角、标点、换行都要与原文一致）")
		return
	}
	fmt.Printf("命中 %d 处。\n", hits)
}

// contextAround 截取引文前后各 pad 个字符，方便人工核对位置。
func contextAround(runes []rune, start, end, pad int) string {
	lo, hi := max(0, start-pad), min(len(runes), end+pad)
	head, tail := "", ""
	if lo < start {
		head = "…"
	}
	if hi > end {
		tail = "…"
	}
	return head + string(runes[lo:start]) + "【" + string(runes[start:end]) + "】" + string(runes[end:hi]) + tail
}

// preview 压掉换行并截断，方便一眼扫过每节开头。
func preview(s string, maxRunes int) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > maxRunes {
		return string(r[:maxRunes]) + "…"
	}
	return s
}

// specQuestion 标注 spec 里的一题。引文是"答案原文"，工具负责把它翻译成 [start,end)。
type specQuestion struct {
	Question string
	Tags     []string
	Note     string
	Quotes   []string
}

// goldSpan 定位结果，同时也是写进 testset v2 的 gold 一项。
// Text 是引文原文：人工核对用，也让 cmd/eval 能自检 span 与正文是否一致。
type goldSpan struct {
	Doc   string `json:"doc"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
}

type outQuestion struct {
	Question string     `json:"question"`
	Tags     []string   `json:"tags,omitempty"`
	Gold     []goldSpan `json:"gold"`
	Note     string     `json:"note,omitempty"`
}

// parseSpec 解析标注 spec。格式（块之间空行随意，行首空白保留在引文里）：
//
//	===Q 什么是 RAG？
//	tags: 字面
//	note: 定义段；旧 gold 含 H1 标题行
//	===S
//	（引文原文，可多行；一块 = 一段 gold span）
//	===S
//	（第二段引文）
//	===Q 下一题
//
// 引文块首尾的空行会被裁掉，中间的空行属于引文本身（多段列表/代码块要靠它）。
func parseSpec(path string) ([]specQuestion, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var (
		out      []specQuestion
		cur      *specQuestion
		quote    []string
		problems []string
	)
	flushQuote := func() {
		if cur == nil || len(quote) == 0 {
			return
		}
		for len(quote) > 0 && strings.TrimSpace(quote[0]) == "" {
			quote = quote[1:]
		}
		for len(quote) > 0 && strings.TrimSpace(quote[len(quote)-1]) == "" {
			quote = quote[:len(quote)-1]
		}
		if len(quote) > 0 {
			cur.Quotes = append(cur.Quotes, strings.Join(quote, "\n"))
		}
		quote = nil
	}
	flushQ := func() {
		flushQuote()
		if cur != nil {
			out = append(out, *cur)
		}
		cur = nil
	}

	for i, raw := range strings.Split(string(b), "\n") {
		line := strings.TrimRight(raw, "\r")
		switch {
		case strings.HasPrefix(line, "#"):
			// 注释行（spec 开头的说明、给标注者看的提醒）
		case strings.HasPrefix(line, "===Q "):
			flushQ()
			cur = &specQuestion{Question: strings.TrimSpace(strings.TrimPrefix(line, "===Q "))}
		case strings.HasPrefix(line, "===S"):
			if cur == nil {
				problems = append(problems, fmt.Sprintf("第 %d 行 ===S 前面没有 ===Q", i+1))
				continue
			}
			flushQuote()
		case cur != nil && len(cur.Quotes) == 0 && len(quote) == 0 && strings.HasPrefix(line, "tags:"):
			for _, t := range strings.Split(strings.TrimPrefix(line, "tags:"), ",") {
				if t = strings.TrimSpace(t); t != "" {
					cur.Tags = append(cur.Tags, t)
				}
			}
		case cur != nil && len(cur.Quotes) == 0 && len(quote) == 0 && strings.HasPrefix(line, "note:"):
			cur.Note = strings.TrimSpace(strings.TrimPrefix(line, "note:"))
		case strings.TrimSpace(line) == "" && len(quote) == 0:
			// 引文块之间的空行：忽略
		default:
			if cur == nil {
				problems = append(problems, fmt.Sprintf("第 %d 行有引文但还没有 ===Q：%s", i+1, preview(line, 30)))
				continue
			}
			quote = append(quote, line)
		}
	}
	flushQ()
	if len(problems) > 0 {
		return nil, fmt.Errorf("spec 格式问题：\n  %s", strings.Join(problems, "\n  "))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s 里没解析出题（===Q 开头？）", path)
	}
	return out, nil
}

// locateQuote 把一段引文定位到唯一一篇文档：0 处 = 抄错了，多处 = 引文不够长（无法唯一定位）。
func locateQuote(docs []*schema.Document, quote string) (*schema.Document, int, error) {
	var (
		hitDoc *schema.Document
		hitAt  int
		hits   []string
	)
	for _, d := range docs {
		for from := 0; from < len(d.Content); {
			i := strings.Index(d.Content[from:], quote)
			if i < 0 {
				break
			}
			byteOff := from + i
			hitAt = utf8.RuneCountInString(d.Content[:byteOff])
			hitDoc = d
			hits = append(hits, fmt.Sprintf("%s[%d,%d)", d.ID, hitAt, hitAt+utf8.RuneCountInString(quote)))
			from = byteOff + len(quote)
		}
	}
	switch len(hits) {
	case 0:
		return nil, 0, fmt.Errorf("语料里找不到该引文（全半角/空格/换行要逐字一致）：%s", preview(quote, 40))
	case 1:
		return hitDoc, hitAt, nil
	default:
		return nil, 0, fmt.Errorf("命中 %d 处，引文不够唯一：%s", len(hits), strings.Join(hits, "、"))
	}
}

// emitTestset 把标注 spec 编译成 testset v2：每题每段引文 → {doc,start,end,text}。
// 引文写错或不够唯一都会在编译期报错，保证 JSON 里的坐标一定指向原文。
func emitTestset(docs []*schema.Document, specPath, outPath string) {
	specs, err := parseSpec(specPath)
	if err != nil {
		log.Fatal(err)
	}

	var (
		problems  []string
		questions []outQuestion
		goldN     int
	)
	for i, s := range specs {
		if len(s.Quotes) == 0 {
			problems = append(problems, fmt.Sprintf("第 %d 题 %q：一段引文都没有", i+1, s.Question))
			continue
		}
		q := outQuestion{Question: s.Question, Tags: s.Tags, Note: s.Note}
		for _, quote := range s.Quotes {
			doc, start, err := locateQuote(docs, quote)
			if err != nil {
				problems = append(problems, fmt.Sprintf("第 %d 题 %q：%v", i+1, s.Question, err))
				continue
			}
			q.Gold = append(q.Gold, goldSpan{Doc: doc.ID, Start: start, End: start + utf8.RuneCountInString(quote), Text: quote})
		}
		goldN += len(q.Gold)
		questions = append(questions, q)
	}

	for _, q := range questions {
		total := 0
		fmt.Printf("%s  【%s】\n", q.Question, strings.Join(q.Tags, " "))
		for _, g := range q.Gold {
			n := utf8.RuneCountInString(g.Text)
			total += n
			fmt.Printf("    %-26s [%6d,%6d) %4d 字  %s\n", g.Doc, g.Start, g.End, n, preview(g.Text, 44))
		}
		fmt.Printf("    gold 合计 %d 字\n\n", total)
	}
	if len(problems) > 0 {
		fmt.Println("定位失败：")
		for _, p := range problems {
			fmt.Println("  " + p)
		}
		log.Fatalf("%d 处没定位好（要求全语料恰好命中一次）；修正引文后重跑", len(problems))
	}

	b, err := json.MarshalIndent(questions, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if outPath == "" {
		fmt.Println(string(b))
		return
	}
	if err := os.WriteFile(outPath, append(b, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("已写入 %s：%d 题 / %d 段 gold\n", outPath, len(questions), goldN)
}
