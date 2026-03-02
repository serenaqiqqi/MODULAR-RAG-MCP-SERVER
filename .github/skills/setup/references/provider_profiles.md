# Provider Profiles Reference

Quick reference for supported provider configurations.

## LLM Providers

### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4o"           # or gpt-4o-mini, gpt-3.5-turbo
  api_key: "<OPENAI_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```
Required fields: `provider`, `model`, `api_key`
Remove/leave empty: `azure_endpoint`, `deployment_name`, `api_version`

### Azure OpenAI
```yaml
llm:
  provider: "azure"
  model: "gpt-4o"
  deployment_name: "<YOUR_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```
Required fields: all shown above

### DeepSeek
```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "<DEEPSEEK_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```

### Ollama (local)
```yaml
llm:
  provider: "ollama"
  model: "llama3"            # or any model pulled via `ollama pull`
  base_url: "http://localhost:11434"
  temperature: 0.0
  max_tokens: 4096
```
No API key required.

## Embedding Providers

### OpenAI
```yaml
embedding:
  provider: "openai"
  model: "text-embedding-ada-002"   # or text-embedding-3-small
  dimensions: 1536                   # 1536 for ada-002, 1536 for 3-small
  api_key: "<OPENAI_API_KEY>"
```

### Azure OpenAI
```yaml
embedding:
  provider: "azure"
  model: "text-embedding-ada-002"
  dimensions: 1536
  deployment_name: "<YOUR_EMBEDDING_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
```

### Ollama
```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  dimensions: 768
  base_url: "http://localhost:11434"
```

## Model → Dimensions Lookup

| Model                      | Dimensions |
|----------------------------|------------|
| text-embedding-ada-002     | 1536       |
| text-embedding-3-small     | 1536       |
| text-embedding-3-large     | 3072       |
| nomic-embed-text (Ollama)  | 768        |
| mxbai-embed-large (Ollama) | 1024       |

## Rerank Providers

### None (disabled)
```yaml
rerank:
  enabled: false
  provider: "none"
  model: ""
  top_k: 5
```

### Cross-Encoder
```yaml
rerank:
  enabled: true
  provider: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5
```

### LLM-based
```yaml
rerank:
  enabled: true
  provider: "llm"
  model: ""  # uses the configured LLM
  top_k: 5
```
