# Guía de Onboarding — Puesta en Marcha para el Cliente

## Introducción

Esta guía está diseñada para que el equipo técnico del cliente pueda desplegar StrategyEngine AI y ejecutar su primer análisis en menos de 30 minutos.

No se requiere experiencia previa con el producto. Solo se necesita acceso a un servidor o máquina con Docker instalado.

---

## Paso 1: Requisitos previos

Antes de empezar, asegúrate de tener lo siguiente:

### Hardware mínimo

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 2 cores | 4 cores |
| Disco | 20 GB libres | 50 GB libres (SSD) |
| Red | Acceso a internet | Acceso a internet |

El acceso a internet es necesario para las llamadas a la API de los modelos de IA. Los datos del cliente nunca salen del servidor — solo se envían los prompts de los agentes.

### Software

| Software | Versión mínima | Cómo verificar |
|----------|---------------|----------------|
| Docker Engine | 20.10+ | `docker --version` |
| Docker Compose | v2+ | `docker compose version` |
| Git | cualquiera | `git --version` |

### Instalación de Docker (si no está instalado)

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Cierra sesión y vuelve a entrar para que el grupo surta efecto
```

**macOS:**
- Descarga Docker Desktop desde https://www.docker.com/products/docker-desktop

**Windows:**
- Descarga Docker Desktop desde https://www.docker.com/products/docker-desktop
- Asegúrate de tener WSL2 habilitado

### API Key de OpenRouter

1. Ve a https://openrouter.ai y crea una cuenta
2. En el panel, ve a **Keys** y haz clic en **Create Key**
3. Copia la clave generada (empieza por `sk-or-v1-...`)
4. Añade créditos a la cuenta (con $10 puedes ejecutar decenas de análisis)

---

## Paso 2: Descargar el producto

```bash
git clone https://github.com/your-org/strategy-engine-ai.git
cd strategy-engine-ai
```

Si recibiste el producto como archivo comprimido en lugar de repositorio:

```bash
unzip strategyengine-ai.zip
cd strategy-engine-ai
```

---

## Paso 3: Configurar

Crea el archivo de configuración a partir de la plantilla:

```bash
cp .env.example .env
```

Abre el archivo `.env` con cualquier editor de texto y pega tu API key:

```env
OPENROUTER_API_KEY=sk-or-v1-tu-clave-aquí
```

Guarda el archivo. Esto es lo único que necesitas configurar.

---

## Paso 4: Arrancar la aplicación

```bash
docker compose up -d
```

La primera vez tardará entre 5 y 15 minutos en descargar e instalar todas las dependencias (frameworks de ML, librerías de visualización, etc.). Las siguientes veces arrancará en segundos.

Verifica que está funcionando:

```bash
docker compose ps
```

Deberías ver algo como:

```
NAME                  STATUS          PORTS
strategyengine-ai     Up (healthy)    0.0.0.0:8501->8501/tcp
```

Si el estado dice `Up (health: starting)`, espera 30 segundos y vuelve a comprobar.

---

## Paso 5: Primer análisis de prueba

### 5.1 Abrir la aplicación

Abre tu navegador y ve a:

```
http://localhost:8501
```

Si el servidor es remoto, sustituye `localhost` por la IP o dominio del servidor.

### 5.2 Preparar un dataset de prueba

Para la primera prueba, usa un CSV sencillo de tu empresa. Requisitos mínimos:

- Formato CSV (separado por comas, punto y coma, o tabulador — se detecta automáticamente)
- Al menos 100 filas
- Al menos 5 columnas
- Una columna que represente lo que quieres predecir o analizar

Si no tienes un CSV a mano, puedes usar cualquier dataset público de Kaggle para verificar que el sistema funciona.

### 5.3 Ejecutar el análisis

1. **Sube el CSV** haciendo clic en el área de carga o arrastrando el archivo
2. **Escribe tu objetivo de negocio** en lenguaje natural. Ejemplos:
   - "Predecir qué clientes van a cancelar su suscripción en los próximos 30 días"
   - "Clasificar los tickets de soporte por prioridad y área responsable"
   - "Estimar el precio óptimo de venta de cada propiedad"
   - "Detectar transacciones fraudulentas en tiempo real"
3. **Haz clic en Iniciar Análisis**

### 5.4 Seguir el progreso

La interfaz muestra en tiempo real:

- La barra de progreso del pipeline (8 etapas)
- El agente que está trabajando en cada momento
- El log de actividad con mensajes de cada agente
- El tiempo transcurrido

Una run típica tarda entre 15 y 30 minutos dependiendo de la complejidad del dataset y los modelos seleccionados.

### 5.5 Revisar los resultados

Cuando termine, verás un dashboard con 7 pestañas:

| Pestaña | Contenido |
|---------|-----------|
| **Estado Inicial** | Tu objetivo de negocio y vista previa del dataset |
| **Auditoría de Datos** | Resumen de calidad de los datos |
| **Estrategia** | Estrategia analítica generada por la IA |
| **Plan de Ejecución** | Contrato técnico con métricas y restricciones |
| **Ingeniería de Datos** | Código de limpieza generado + datos procesados |
| **Modelo ML** | Código del modelo + logs de ejecución |
| **Informe Ejecutivo** | Reporte PDF descargable con decisión, gráficos y recomendaciones |

### 5.6 Descargar el informe

En la pestaña **Informe Ejecutivo**, haz clic en **Descargar PDF** para obtener el reporte ejecutivo listo para compartir con dirección.

---

## Paso 6: Conectar tu CRM (opcional)

Si quieres analizar datos directamente desde tu CRM sin exportar CSVs manualmente:

1. En el panel lateral, haz clic en **Conectar CRM**
2. Selecciona tu proveedor (HubSpot, Salesforce o Dynamics 365)
3. Introduce las credenciales de conexión
4. Haz clic en **Probar conexión** para verificar
5. Selecciona el objeto y las filas a importar

Las credenciales se almacenan cifradas localmente en el servidor. Nunca se envían a servicios externos.

Para instrucciones detalladas de cada CRM, consulta [03_INTEGRACION_CRM.md](03_INTEGRACION_CRM.md).

---

## Verificación final

Usa esta checklist para confirmar que todo funciona correctamente:

- [ ] `docker compose ps` muestra el contenedor como `Up (healthy)`
- [ ] La interfaz web carga en `http://localhost:8501`
- [ ] Puedes subir un CSV y se muestra la vista previa
- [ ] Un análisis de prueba completa sin errores
- [ ] El informe ejecutivo PDF se genera y se puede descargar
- [ ] (Opcional) La conexión CRM funciona y puedes importar datos

---

## Operaciones básicas

### Detener la aplicación

```bash
docker compose down
```

### Reiniciar después de un corte

```bash
docker compose up -d
```

Los datos y el historial de runs se conservan automáticamente.

### Ver logs si algo falla

```bash
docker compose logs -f app
```

### Actualizar a una nueva versión

```bash
git pull origin main
docker compose up -d --build
```

---

## Soporte

Si encuentras cualquier problema durante la puesta en marcha:

- **Email**: contacto@strategyengine.ai
- **Documentación completa**: ver carpeta `documentacion/` del producto
- **Guía del Sandbox Gateway**: [04_SANDBOX_GATEWAY.md](04_SANDBOX_GATEWAY.md) (para ejecución en infraestructura propia)
- **Configuración de modelos**: [05_CONFIGURACION_MODELOS.md](05_CONFIGURACION_MODELOS.md) (para ajustar coste/calidad)
