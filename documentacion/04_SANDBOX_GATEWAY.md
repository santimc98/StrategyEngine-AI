# Guía de Integración del Sandbox Gateway

## Objetivo

Este documento explica cómo una empresa puede conectar su propio sandbox de ejecución a StrategyEngine AI sin modificar el core del producto.

El producto soporta dos modos de ejecución:

- `local`: la run se ejecuta en la misma máquina donde corre la app
- `remote`: la run se ejecuta a través de un sandbox gateway HTTP alojado por la propia empresa

La configuración recomendada para clientes enterprise es usar `remote` y desplegar ese gateway dentro de su propia nube o infraestructura interna. Así, el coste de compute, almacenamiento, red y seguridad queda dentro del entorno del cliente.

## Modelo de producto

La app **no** se integra por separado con Google Cloud, Azure, AWS ni con ningún sandbox específico de proveedor.

En su lugar, se integra con un único protocolo:

- un gateway HTTP de sandbox alojado por la empresa que implemente una API pequeña y estable

Esto implica:

- la UI es la misma para cualquier empresa
- el grafo y los agentes no necesitan cambios por cliente
- la empresa puede implementar ese gateway sobre Cloud Run, GKE, Batch, Kubernetes, VMs, contenedores o cualquier orquestador interno

## Qué configura el cliente en la UI

La UI expone estos campos para el provider `remote`:

- `Endpoint del gateway`
- `API key`
- `Auth scheme`
- `Auth header`
- `Workspace`
- `Project`
- `Hint de infraestructura`
- `Timeout HTTP`

Esa configuración se guarda localmente y se pasa a cada run como `sandbox_config`.

## API requerida del gateway

El gateway debe exponer los siguientes endpoints.

### 1. Health Check

`GET /health`

Objetivo:
- usado por la UI para validar que el gateway remoto es accesible

Respuesta esperada:

```json
{
  "ok": true
}
```

Cualquier respuesta `2xx` se considera saludable por el cliente actual.

### 2. Crear sesión

`POST /sessions`

Objetivo:
- crear una sesión aislada de ejecución para una run

Cuerpo de la request:

```json
{
  "workspace_id": "analytics-prod",
  "project": "customer-project",
  "provider_hint": "gcp",
  "metadata": {
    "optional": "free-form metadata"
  }
}
```

Respuesta requerida:

```json
{
  "session_id": "sess-123"
}
```

Alternativa aceptada:
- `id` puede devolverse en lugar de `session_id`

### 3. Escribir archivo

`POST /sessions/{session_id}/files/write`

Objetivo:
- subir al sandbox archivos de entrada, scripts, manifests, contratos y artefactos auxiliares

Cuerpo de la request:

```json
{
  "path": "/home/user/run/abc/data/raw.csv",
  "content_base64": "SGVsbG8="
}
```

Respuesta esperada:

```json
{
  "ok": true
}
```

Si falla la escritura:

```json
{
  "ok": false,
  "error": "reason"
}
```

### 4. Leer archivo

`GET /sessions/{session_id}/files/read?path=/home/user/run/abc/data/output.csv`

Objetivo:
- descargar desde el sandbox los artefactos generados por la run

Respuesta requerida:

```json
{
  "content_base64": "SGVsbG8="
}
```

### 5. Ejecutar comando

`POST /sessions/{session_id}/commands/run`

Objetivo:
- ejecutar comandos shell dentro de la sesión del sandbox

Cuerpo de la request:

```json
{
  "cmd": "python /home/user/run/abc/ml_engineer/script.py",
  "timeout": 600
}
```

Respuesta requerida:

```json
{
  "stdout": "logs...",
  "stderr": "",
  "exit_code": 0
}
```

### 6. Cerrar sesión

`POST /sessions/{session_id}/close`

Objetivo:
- liberar los recursos del sandbox cuando termina la run

Respuesta esperada:

```json
{
  "ok": true
}
```

## Autenticación

El cliente soporta autenticación genérica basada en cabeceras HTTP.

Campos de la UI:

- `API key`
- `Auth scheme`
- `Auth header`

Comportamiento:

- si `Auth scheme = Bearer` y `Auth header = Authorization`, las requests incluyen:

```text
Authorization: Bearer <api_key>
```

- si `Auth scheme = none`, la clave se envía directamente:

```text
<Auth header>: <api_key>
```

Esto se ha hecho así a propósito para que cada empresa pueda adaptarlo a su convención de autenticación sin cambiar el producto.

## Modelo de sesión

Cada run debe mapearse a una sesión aislada.

Propiedades recomendadas:

- aislamiento de filesystem por sesión
- aislamiento de procesos por sesión
- limpieza explícita en `/close`
- enforcement de timeouts
- cuotas de recursos
- auditoría y logging

El producto asume que puede:

- subir archivos
- ejecutar comandos varias veces dentro de la misma sesión
- descargar artefactos
- cerrar la sesión cuando termina

## Semántica de archivos

La app escribe archivos usando rutas de sandbox como:

- `/home/user/run/<run_id>/...`

Tu gateway no necesita conservar esa ruta literalmente en disco, pero sí debe conservar su identidad lógica dentro de la sesión.

En la práctica:

- mapea la ruta entrante a tu propio layout interno
- mantén estable el contrato de rutas dentro de la sesión

## Requisitos de manejo de errores

Para que el sistema funcione bien, el gateway debería:

- devolver respuestas JSON deterministas
- preservar `stdout`, `stderr` y `exit_code`
- devolver `non-2xx` o `ok=false` estructurado cuando falle una operación
- aplicar timeouts del lado servidor
- evitar truncar inesperadamente la salida de comandos

Recomendado:

- incluir request IDs en los logs del gateway
- incluir session IDs en todos los logs internos
- conservar logs de ejecución para auditoría

## Recomendaciones de seguridad

Controles mínimos recomendados:

- TLS habilitado
- credenciales de vida corta si es posible
- allowlisting de red si el gateway es privado
- aislamiento estricto por sesión
- no persistir credenciales dentro de las sesiones
- política de retención de artefactos
- limpieza explícita de sesiones

Muy recomendable:

- desplegar el gateway en la cuenta cloud del cliente
- ejecutar los workers reales del sandbox dentro del mismo boundary de seguridad
- no exponer directamente APIs de VM, Kubernetes o infraestructura al producto

## Despliegue de referencia en Google Cloud

Si el cliente usa Google Cloud, el patrón recomendado es:

1. desplegar un servicio HTTP pequeño que haga de gateway
2. autenticar las requests de StrategyEngine AI hacia ese gateway
3. dejar que el gateway cree y gestione el backend real de ejecución

Backends posibles detrás del gateway:

- Cloud Run Jobs
- GKE
- Batch
- Compute Engine
- un servicio interno de sandbox que ya opere el cliente

Punto de partida recomendado en GCP:

- gateway en Cloud Run
- backend de ejecución en Cloud Run Jobs o GKE
- logs en Cloud Logging
- artefactos en Cloud Storage si hace falta

Con ese modelo:

- la empresa paga el compute y el storage en su propia cuenta GCP
- StrategyEngine AI solo consume el gateway del cliente

## Ejemplo mínimo de gateway

El gateway puede implementarse en cualquier lenguaje. El único requisito es respetar el contrato de API anterior.

Flujo de alto nivel:

1. recibir `/sessions`
2. crear un contexto aislado de ejecución
3. guardar metadata de sesión
4. aceptar subidas de archivos
5. ejecutar comandos
6. exponer archivos generados
7. limpiar recursos en `/close`

## Garantías del producto

Con este diseño:

- las empresas pequeñas pueden usar `local`
- los clientes enterprise pueden usar `remote`
- no hacen falta cambios por empresa dentro del grafo
- no hay lock-in del producto con un proveedor cloud concreto
- el coste del sandbox queda en la infraestructura del cliente

## Checklist de integración

Antes de activar `remote` en producción, verifica:

1. `/health` responde `2xx`
2. `/sessions` devuelve `session_id`
3. la subida de archivos funciona con payloads base64
4. la ejecución de comandos devuelve `stdout/stderr/exit_code` completos
5. la descarga de archivos devuelve base64 válido
6. `/close` libera recursos
7. el aislamiento por sesión está garantizado
8. la política de timeouts está aplicada
9. los logs pueden trazarse por `session_id`
10. endpoint y credenciales funcionan desde el host donde corre StrategyEngine AI
