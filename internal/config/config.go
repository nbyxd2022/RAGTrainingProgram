package config

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	ArkAPIKey         string
	ArkChatModel      string
	ArkEmbeddingModel string
}

func Load() (*Config, error) {
	// .env 不存在时忽略，允许直接用环境变量注入
	_ = godotenv.Load()

	cfg := &Config{
		ArkAPIKey:         os.Getenv("ARK_API_KEY"),
		ArkChatModel:      os.Getenv("ARK_CHAT_MODEL"),
		ArkEmbeddingModel: os.Getenv("ARK_EMBEDDING_MODEL"),
	}

	var missing []string
	if cfg.ArkAPIKey == "" {
		missing = append(missing, "ARK_API_KEY")
	}
	if cfg.ArkChatModel == "" {
		missing = append(missing, "ARK_CHAT_MODEL")
	}
	if cfg.ArkEmbeddingModel == "" {
		missing = append(missing, "ARK_EMBEDDING_MODEL")
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("缺少必填配置 %v（复制 .env.example 为 .env 后填写）", missing)
	}
	return cfg, nil
}
