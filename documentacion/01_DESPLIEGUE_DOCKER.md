# Guía de Despliegue con Docker

## Requisitos previos

- Docker Engine 20.10+ instalado
- Docker Compose v2 instalado
- Una API key de OpenRouter (https://openrouter.ai)
- Mínimo 8 GB de RAM disponible para el contenedor

## Despliegue rápido

### 1. Clonar el repositorio

```bash
git clone https://github.com/your-org/strategy-engine-ai.git
cd strategy-engine-ai
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita el archivo `.env` y añade tu API key de OpenRouter:

```env
OPENROUTER_API_KEY=sk-or-v1-tu-clave-aquí
```

### 3. Construir y levantar

```bash
docker compose up -d
```

La primera vez tardará varios minutos en descargar las dependencias de ML (PyTorch, TensorFlow, scikit-learn, etc.).

### 4. Acceder a la aplicación

Abre en tu navegador:

```
http://localhost:8501
```

## Estructura de archivos

```
strategy-engine-ai/
  Dockerfile              # Imagen de la aplicación
  docker-compose.yml      # Orquestación del servicio
  .env                    # Variables de entorno (no commitear)
  .env.example            # Plantilla de variables
  .dockerignore           # Exclusiones del build
```

## Volúmenes persistentes

El `docker-compose.yml` monta dos volúmenes para que los datos sobrevivan a reinicios del contenedor:

| Volumen | Ruta en contenedor | Contenido |
|---------|-------------------|-----------|
| `./runs` | `/app/runs` | Historial de ejecuciones, artefactos, logs |
| `./data` | `/app/data` | Credenciales CRM cifradas, configuración |

Si necesitas hacer backup, copia estas dos carpetas.

## Verificar que funciona

### Health check

```bash
curl http://localhost:8501/_stcore/health
```

Respuesta esperada: `ok`

### Logs del contenedor

```bash
docker compose logs -f app
```

### Estado del contenedor

```bash
docker compose ps
```

## Detener y reiniciar

```bash
# Detener
docker compose down

# Reiniciar
docker compose up -d

# Reconstruir después de una actualización
docker compose up -d --build
```

## Configuración avanzada

### Cambiar el puerto

Si el puerto 8501 ya está en uso, edita `docker-compose.yml`:

```yaml
ports:
  - "9090:8501"  # Acceder en http://localhost:9090
```

### Limitar recursos

Para controlar el consumo de RAM y CPU del contenedor:

```yaml
services:
  app:
    # ... configuración existente ...
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: "4"
```

### Usar un modelo LLM específico

Los modelos se configuran desde la interfaz web en el panel lateral. La selección persiste automáticamente en el volumen `./data`.

### Variables de entorno opcionales

| Variable | Valor por defecto | Descripción |
|----------|------------------|-------------|
| `OPENROUTER_API_KEY` | -- | Clave de OpenRouter (obligatoria) |
| `OPENROUTER_TIMEOUT_SECONDS` | `120` | Timeout por llamada LLM |
| `SANDBOX_GATEWAY_URL` | -- | URL del gateway remoto (ver guía de Sandbox Gateway) |

## Actualizar a una nueva versión

```bash
git pull origin main
docker compose up -d --build
```

Los volúmenes `runs/` y `data/` no se ven afectados por la actualización.

## Solución de problemas

### El contenedor no arranca

```bash
docker compose logs app
```

Causas comunes:
- Puerto 8501 ya en uso por otro proceso
- Falta la variable `OPENROUTER_API_KEY` en `.env`
- Memoria insuficiente para instalar las dependencias ML

### Las runs fallan con errores de timeout

Aumenta el timeout de OpenRouter:

```yaml
environment:
  - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
  - OPENROUTER_TIMEOUT_SECONDS=300
```

### Los datos se pierden al reiniciar

Verifica que los volúmenes están montados correctamente:

```bash
docker compose config | grep volumes -A 5
```

Los directorios `./runs` y `./data` deben existir en el host.
