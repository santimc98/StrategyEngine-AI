# Guía de Configuración de Modelos LLM

## Visión general

StrategyEngine AI utiliza modelos de lenguaje (LLMs) a través de OpenRouter, una pasarela unificada que da acceso a múltiples proveedores (Google, OpenAI, Anthropic, etc.) con una sola API key.

Cada agente del pipeline puede configurarse con un modelo diferente según las necesidades de coste, velocidad y calidad.

## Requisito: API Key de OpenRouter

1. Crea una cuenta en https://openrouter.ai
2. Ve a **Keys** y genera una nueva API key
3. Configura la key en tu `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-tu-clave-aquí
```

Con esta única clave, todos los agentes tienen acceso a todos los modelos disponibles en OpenRouter.

## Configuración desde la interfaz

1. Abre la aplicación web
2. En el panel lateral izquierdo, busca la sección **Configuración de Modelos**
3. Para cada agente configurable, selecciona el modelo deseado del dropdown
4. Los cambios se guardan automáticamente y persisten entre sesiones

### Agentes configurables

| Agente | Rol | Modelo recomendado |
|--------|-----|-------------------|
| **Strategist** | Genera estrategia y hipótesis | GLM-5 (coste-efectivo) o Claude Opus (máxima calidad) |
| **Data Engineer** | Genera código de limpieza y ETL | GLM-5 o GPT Codex (mejor generación de código) |
| **ML Engineer** | Genera código de modelo ML | GPT Codex (mejor para código complejo) |
| **Model Analyst** | Análisis de rendimiento del modelo | Hereda del Strategist |

Los agentes restantes (Steward, Planner, Reviewers, Translator, etc.) usan Gemini por defecto y no son configurables desde la UI — están optimizados para su tarea específica.

## Modelos disponibles

### Presets incluidos

| Preset | Modelo | Proveedor | Uso recomendado |
|--------|--------|-----------|----------------|
| GLM-5 | `thudm/glm-4-plus` | Zhipu | Balance coste/calidad (default) |
| Kimi K2.5 | `moonshotai/kimi-k2.5` | Moonshot | Alternativa económica |
| Minimax M-2.5 | `minimax/minimax-m1-80k` | Minimax | Contexto largo |
| DeepSeek V3.2 | `deepseek/deepseek-chat-v3-0324` | DeepSeek | Razonamiento técnico |
| Claude Opus 4.6 | `anthropic/claude-opus-4-6` | Anthropic | Máxima calidad de razonamiento |
| GPT-5.3 Codex | `openai/gpt-5.3-codex` | OpenAI | Generación de código premium |
| GPT-5.4 | `openai/gpt-5.4` | OpenAI | Último modelo de OpenAI |

### Modelo custom

Si necesitas usar un modelo que no está en los presets:

1. Selecciona **Custom** en el dropdown
2. Introduce el model ID de OpenRouter (ej: `google/gemini-2.5-pro`)
3. Consulta el catálogo completo en https://openrouter.ai/models

## Costes estimados

El coste por run varía según los modelos seleccionados y la complejidad del análisis.

| Configuración | Coste estimado por run | Caso de uso |
|--------------|----------------------|-------------|
| Todos GLM-5 | ~$0.30 - $0.80 | Desarrollo, pruebas, datasets simples |
| Mix GLM-5 + Codex | ~$0.50 - $1.50 | Producción estándar |
| Todos premium (Opus/Codex) | ~$2.00 - $5.00 | Análisis críticos, datasets complejos |

Factores que afectan el coste:
- Número de iteraciones de mejora del modelo (1-12 rondas)
- Tamaño del dataset (más columnas = prompts más largos)
- Complejidad de la estrategia (más técnicas = más código generado)
- Reintentos por errores de ejecución

## Recomendaciones por tipo de cliente

### Startup / equipo pequeño
- Usa GLM-5 para todo
- Coste mínimo, resultados buenos para la mayoría de casos

### Empresa mediana
- Strategist: GLM-5
- ML Engineer: GPT Codex (mejor código)
- Data Engineer: GLM-5
- Balance óptimo de coste y calidad

### Enterprise / análisis críticos
- Strategist: Claude Opus (mejor razonamiento estratégico)
- ML Engineer: GPT Codex
- Data Engineer: GPT Codex
- Máxima calidad, coste asumible para decisiones importantes

## Fallback automático

Si un modelo falla (timeout, rate-limit, error del proveedor), el sistema intenta automáticamente con modelos alternativos de la cadena de fallback. Esto garantiza que las runs no se interrumpan por problemas temporales de un proveedor específico.
