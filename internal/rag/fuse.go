package rag

import (
	"sort"

	"github.com/cloudwego/eino/schema"
)

// FuseRRF 用 Reciprocal Rank Fusion 融合多路检索结果 —— 步骤 2A 任务 3，由你实现。
//
// 每路结果是一个按相关性降序的 ID 列表。融合规则：
// 某条结果在某一路上排第 r 名（r 从 1 开始），就贡献 1/(k+r) 分，各路的贡献相加。
//
// 要求：
//  1. r 从 1 开始计；k 用调用方传入的值（惯例 60）
//  2. 同一条 ID 出现在多路上，分数累加
//  3. 返回所有出现过的 ID，按总分降序
//  4. 总分并列时按 ID 升序兜底 —— 保证每次运行结果一致，方便对比实验
//  5. 不要修改传入的 lists（排序要排自己的副本）
//
// 提示：需要 map[string]float64 累计分数，收集好 key 之后再用 sort.Slice 排序。
func FuseRRF(lists [][]string, k int) []string {

	score := make(map[string]float64)
	for _, list := range lists {
		for i, id := range list {
			r := i + 1
			score[id] += 1.0 / float64(k+r)
		}
	}
	//注意 map 的遍历顺序是随机的，要对map进行排序，标准做法就是下面的做法
	// 先把map的key抽出来放到一个切片里，然后对key进行排序，这里说的对key排序不一定是a按照key的值继续排序，我们也可以用下面的方法按照value来排序
	ids := make([]string, 0, len(score))
	for key := range score {
		ids = append(ids, key)
	}

	sort.Slice(ids, func(i int, j int) bool {
		if score[ids[i]] > score[ids[j]] {
			return true
		} else if score[ids[i]] < score[ids[j]] {
			return false
		} else {
			return ids[i] < ids[j]
		}

	})

	return ids
}

// FuseDocuments 把多路检索到的文档融合成一路。
// 做法：先按 ID 建立"融合池"（取第一次出现的文档作代表），跑 RRF 得到排序，再按序重建文档。
// 重建时复制元数据并写入真实的 RRF 总分，方便在对比输出里看到"两路都命中"的加成。
func FuseDocuments(lists [][]*schema.Document, k int) []*schema.Document {
	byID := make(map[string]*schema.Document)
	idLists := make([][]string, len(lists))
	for i, list := range lists {
		ids := make([]string, 0, len(list))
		for _, d := range list {
			if _, ok := byID[d.ID]; !ok {
				byID[d.ID] = d
			}
			ids = append(ids, d.ID)
		}
		idLists[i] = ids
	}

	fused := FuseRRF(idLists, k)

	// 重算总分用于展示（与 FuseRRF 同一公式，只影响 meta["score"]，不影响排序）
	totals := make(map[string]float64, len(fused))
	for _, ids := range idLists {
		for r, id := range ids {
			totals[id] += 1.0 / float64(k+r+1)
		}
	}

	out := make([]*schema.Document, 0, len(fused))
	for _, id := range fused {
		d := byID[id]
		meta := make(map[string]any, len(d.MetaData)+1)
		for mk, mv := range d.MetaData {
			meta[mk] = mv
		}
		meta["score"] = totals[id]
		out = append(out, &schema.Document{ID: d.ID, Content: d.Content, MetaData: meta})
	}
	return out
}
