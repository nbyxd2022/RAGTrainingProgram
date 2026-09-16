package rag

import (
	"fmt"
	"strings"

	"github.com/cloudwego/eino/schema"
)

// BuildMessages 把问题和检索到的资料组装成发给模型的对话 —— 步骤 5 由你实现。
//
// 要求：
//  1. 返回两条消息：系统消息（定义助手的角色和行为边界）+ 用户消息（资料 + 问题）
//  2. 系统消息里至少写清楚两件事：
//     - 只能依据给定资料回答
//     - 资料不足以回答时，明确说"资料里没有相关内容"，不要编造
//  3. 用户消息里每段资料要带编号和来源，例如：
//     [1] 来源：corpus/01-rag.md
//     <正文>
//     编号方便模型引用（"根据[1]"），来源方便你事后核对
//  4. 不要把 score 放进提示词——那是给工程用的元数据，对模型只是噪声
//  5. 资料和问题之间的边界要清楚（标题、空行、分隔线都行），模型要能一眼分清
//     哪些是资料、哪句才是要回答的问题
func BuildMessages(question string, docs []*schema.Document) []*schema.Message {

	msgs := []*schema.Message{
		schema.SystemMessage("你是一个知识渊博的助手，只能依据给定资料回答问题。如果你引用了资料，必须在引用后标明资料的序号和资料来源。资料不足以回答时，明确说“资料里没有相关内容”，不要编造。"),
	}

	var sb strings.Builder
	sb.WriteString("以下是你回答前要参考的资料：------------------------------------------------------\n")

	for i, doc := range docs {
		fmt.Fprintf(&sb, "[%d] 来源：%s\n%s\n", i+1, doc.MetaData["source"].(string), doc.Content)
	}
	sb.WriteString("------------------------------------------------------\n")
	fmt.Fprintf(&sb, "以下是用户提出的问题：\n%s\n", question)
	msgs = append(msgs, schema.UserMessage(sb.String()))

	return msgs
}
