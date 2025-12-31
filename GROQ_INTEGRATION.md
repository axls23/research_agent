# Groq Open-Source Models Integration

## 🚀 Overview

This document describes the integration of open-source AI models using Groq as the inference provider in the Research Agent system. Groq's Language Processing Unit (LPU) architecture provides ultra-low latency and high throughput, making it ideal for real-time AI applications.

## 📋 Available Models

### Supported Open-Source Models

| Model | Category | Size | Context Length | Best For |
|-------|----------|------|----------------|----------|
| `llama-3.1-70b-instruct` | Llama | 70B | 131K | Complex reasoning, research analysis |
| `llama-3.1-8b-instruct` | Llama | 8B | 131K | General purpose, fast responses |
| `mixtral-8x7b-32768` | Mixtral | 8x7B | 32K | Long context, multilingual |
| `gemma-7b-it` | Gemma | 7B | 8K | General purpose, efficient |
| `codellama-70b-instruct` | CodeLlama | 70B | 100K | Code generation, programming |
| `deepseek-r1-distill-1.5b` | DeepSeek | 1.5B | 32K | Reasoning, mathematics |
| `qwen-32b-instruct` | Qwen | 32B | 32K | Multilingual, research |
| `mistral-7b-instruct` | Mistral | 7B | 32K | General purpose, analysis |

### Model Categories

- **Llama Series**: Meta's open-source models, excellent for general tasks
- **Mixtral Series**: Mixture of Experts models for specialized tasks
- **Gemma Series**: Google's efficient open-source models
- **CodeLlama Series**: Specialized for code generation and programming
- **DeepSeek Series**: Reasoning-focused models for complex problems
- **Qwen Series**: Multilingual models with strong capabilities
- **Mistral Series**: Efficient models for various applications

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install groq>=0.4.0
```

### 2. Set Environment Variables

```bash
export GROQ_API_KEY="your-groq-api-key-here"
export GROQ_ENABLED="true"
export GROQ_DEFAULT_MODEL="llama-3.1-8b-instruct"
```

### 3. Get Groq API Key

1. Visit [GroqCloud Console](https://console.groq.com/)
2. Sign up for an account
3. Generate an API key
4. Set the `GROQ_API_KEY` environment variable

## 📖 Usage Examples

### Basic Text Generation

```python
from research_agent.utils.groq_models import GroqModelClient

# Initialize client
client = GroqModelClient()

# Generate text
response = await client.generate_text(
    "Explain quantum computing in simple terms",
    model="llama-3.1-8b-instruct"
)
print(response)
```

### Research Topic Analysis

```python
# Analyze a research topic
analysis = await client.analyze_research_topic(
    "Machine Learning in Healthcare",
    model="llama-3.1-70b-instruct"
)
print(analysis['analysis'])
```

### Code Generation

```python
# Generate code
code_result = await client.generate_code(
    "A function to calculate fibonacci numbers",
    language="python",
    model="codellama-70b-instruct"
)
print(code_result['code'])
```

### Document Summarization

```python
# Summarize a document
summary = await client.summarize_document(
    document_content,
    max_length=500,
    model="llama-3.1-8b-instruct"
)
print(summary['summary'])
```

### Using Convenience Functions

```python
from research_agent.utils.groq_models import quick_generate, analyze_topic

# Quick generation
response = await quick_generate("What is AI?", model="llama-3.1-8b-instruct")

# Topic analysis
analysis = await analyze_topic("Climate Change Research")
```

## 🤖 Agent Integration

### Literature Review Agent

```python
from research_agent.agents.literature_review_agent import LiteratureReviewAgent

agent = LiteratureReviewAgent()

# Formulate search queries
result = await agent.execute({
    "action": "formulate_search_query",
    "topic": "Deep Learning in Medical Imaging",
    "research_goals": ["accuracy", "efficiency"]
})

# Analyze papers
result = await agent.execute({
    "action": "analyze_paper",
    "paper": {
        "title": "Deep Learning for Medical Image Analysis",
        "abstract": "This paper presents...",
        "authors": "Smith et al."
    }
})
```

### Writing Assistant Agent

```python
from research_agent.agents.writing_assistant_agent import WritingAssistantAgent

agent = WritingAssistantAgent()

# Generate outline
result = await agent.execute({
    "action": "generate_outline",
    "topic": "The Future of AI",
    "outline_type": "research_paper"
})

# Synthesize literature
result = await agent.execute({
    "action": "synthesize_literature",
    "papers": [paper1, paper2, paper3],
    "topic": "Machine Learning Applications"
})
```

## ⚙️ Configuration

### System Configuration

The Groq integration is configured in `config/config.yaml`:

```yaml
llm:
  provider: "groq"  # or "google" | "openai" | "anthropic"
  groq_enabled: true
  groq_api_key: null  # Set via GROQ_API_KEY env var
  groq_default_model: "llama-3.1-8b-instruct"
  groq_models:
    - "llama-3.1-70b-instruct"
    - "llama-3.1-8b-instruct"
    - "mixtral-8x7b-32768"
    - "gemma-7b-it"
    - "codellama-70b-instruct"
    - "deepseek-r1-distill-1.5b"
    - "qwen-32b-instruct"
    - "mistral-7b-instruct"
```

### Model Selection Strategy

The system uses intelligent model selection based on task requirements:

- **Complex Reasoning**: `llama-3.1-70b-instruct`
- **Code Generation**: `codellama-70b-instruct`
- **Fast Responses**: `llama-3.1-8b-instruct`
- **Long Context**: `mixtral-8x7b-32768`
- **Multilingual**: `qwen-32b-instruct`

## 🧪 Testing

### Run Integration Tests

```bash
python test_groq_integration.py
```

### Test Individual Components

```python
# Test basic functionality
from research_agent.utils.groq_models import test_groq_models
await test_groq_models()

# Test agents
from research_agent.agents.literature_review_agent import LiteratureReviewAgent
agent = LiteratureReviewAgent()
result = await agent.execute({"action": "formulate_search_query", "topic": "AI"})
```

## 📊 Performance Characteristics

### Speed & Latency

- **Ultra-low latency**: Groq's LPU architecture provides responses in milliseconds
- **High throughput**: Process hundreds of tokens per second
- **Real-time applications**: Suitable for interactive AI applications

### Model Performance

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| Llama 3.1 70B | Medium | High | Complex reasoning |
| Llama 3.1 8B | Fast | Good | General purpose |
| Mixtral 8x7B | Medium | High | Long context |
| CodeLlama 70B | Medium | High | Code generation |
| DeepSeek R1 | Fast | Good | Reasoning |

## 🔧 Advanced Features

### Streaming Responses

```python
# Stream responses for real-time applications
response = await client.generate_text(
    "Write a story about AI",
    model="llama-3.1-8b-instruct",
    stream=True
)

for chunk in response['chunks']:
    print(chunk, end='', flush=True)
```

### Custom Model Configuration

```python
# Use custom parameters
response = await client.generate_text(
    "Explain machine learning",
    model="llama-3.1-70b-instruct",
    temperature=0.3,
    max_tokens=1000,
    top_p=0.9
)
```

### Model Registry

```python
from research_agent.utils.groq_models import GroqModelRegistry

registry = GroqModelRegistry()

# List models by category
llama_models = registry.list_models_by_category("llama")

# List models by capability
code_models = registry.list_models_by_capability("code_generation")

# Get model information
model_info = registry.get_model_config("llama-3.1-70b-instruct")
print(f"Max tokens: {model_info.max_tokens}")
print(f"Capabilities: {model_info.capabilities}")
```

## 🚨 Error Handling

### Common Issues

1. **API Key Not Set**
   ```python
   # Error: ValueError: Groq API key is required
   # Solution: Set GROQ_API_KEY environment variable
   ```

2. **Model Not Available**
   ```python
   # Error: Model not found in registry
   # Solution: Use a supported model name
   ```

3. **Rate Limiting**
   ```python
   # Error: Rate limit exceeded
   # Solution: Implement retry logic with exponential backoff
   ```

### Error Recovery

```python
try:
    response = await client.generate_text("Hello", model="llama-3.1-8b-instruct")
except Exception as e:
    logger.error(f"Groq API error: {e}")
    # Fallback to alternative model or provider
    response = await fallback_generate("Hello")
```

## 📈 Monitoring & Logging

### Usage Tracking

The system automatically tracks:
- Model usage and performance
- Token consumption
- Response times
- Error rates

### Logging Configuration

```python
import logging

# Enable detailed logging
logging.getLogger("research_agent.utils.groq_models").setLevel(logging.DEBUG)
```

## 🔄 Integration with Existing System

### Workflow Integration

The Groq models integrate seamlessly with the existing research agent workflow:

1. **Project Creation**: Uses Groq for initial analysis
2. **Literature Review**: Groq-powered paper analysis
3. **Writing Assistance**: Groq-based content generation
4. **Knowledge Synthesis**: Groq for complex reasoning tasks

### Fallback Strategy

The system implements intelligent fallbacks:
1. Primary: Groq models
2. Fallback 1: Google Gemini
3. Fallback 2: OpenAI GPT-4
4. Fallback 3: Local models (if available)

## 🎯 Best Practices

### Model Selection

- Use **Llama 3.1 70B** for complex reasoning tasks
- Use **Llama 3.1 8B** for fast, general-purpose tasks
- Use **CodeLlama 70B** for programming and code generation
- Use **Mixtral 8x7B** for long-context applications

### Performance Optimization

- Batch requests when possible
- Use appropriate model sizes for tasks
- Implement caching for repeated queries
- Monitor token usage and costs

### Error Handling

- Always implement try-catch blocks
- Use fallback models for critical tasks
- Log errors for debugging
- Implement retry logic for transient failures

## 📚 Additional Resources

- [Groq Documentation](https://console.groq.com/docs)
- [Groq Model List](https://console.groq.com/docs/models)
- [Research Agent Documentation](./README.md)
- [API Reference](./docs/api-reference.md)

## 🤝 Contributing

To add new models or improve the integration:

1. Update `GroqModelRegistry` with new model configurations
2. Add model-specific optimizations in `GroqModelClient`
3. Update agent implementations to use new capabilities
4. Add tests for new functionality
5. Update documentation

## 📄 License

This integration is part of the Research Agent project and follows the same license terms.
